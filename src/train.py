import os
import argparse
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import CompleteHierarchicalDrumDataset
from model import HierarchicalDrumModel

# -------------------- Collate Function --------------------
def collate_fn(batch):
    return {
        "step": nn.utils.rnn.pad_sequence([d["step"] for d in batch], batch_first=True),
        "bar": nn.utils.rnn.pad_sequence([d["bar"] for d in batch], batch_first=True),
        "phrase": nn.utils.rnn.pad_sequence([d["phrase"] for d in batch], batch_first=True),
        "metrics": [d["metrics"] for d in batch]
    }

# -------------------- RL Reward Computation --------------------
def compute_reward(logits, metrics, device):
    step_metrics = metrics['step']
    dense_frac = torch.tensor(step_metrics['dense_frac'], device=device)
    sync = torch.tensor(step_metrics['sync'], device=device)
    ioi_var = torch.tensor(step_metrics['ioi_stats'][:, 1].mean(), device=device)
    
    weights = torch.tensor([dense_frac, sync, ioi_var], device=device)
    weights = weights / (weights.sum() + 1e-6)
    reward = (weights * torch.tensor([dense_frac, sync, 1.0 / (ioi_var + 1e-6)], device=device)).sum()
    return reward, weights

# -------------------- Checkpointing --------------------
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint['epoch']

# -------------------- Training Loop --------------------
def train(rank, world_size, args):
    # -------------------- Device & DDP Setup --------------------
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
        print(f"[Rank {local_rank}] DDP initialized with {world_size} GPUs")
    else:
        print("[DEBUG] Single GPU or CPU mode")

    # -------------------- Dataset & DataLoader --------------------
    dataset = CompleteHierarchicalDrumDataset(args.data_dir, seq_len=args.seq_len, augment=args.augment)
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn
    )

    # -------------------- Model --------------------
    model = HierarchicalDrumModel(
        num_drums=args.num_drums,
        step_hidden_dim=args.d_model,
        bar_hidden_dim=args.d_model,
        phrase_hidden_dim=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        print(f"[Rank {local_rank}] Model wrapped in DDP")

    # -------------------- Optimizer, Scheduler & AMP --------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device_type='cuda' if device.type == 'cuda' else 'cpu')

    bce_loss = nn.BCELoss()
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        print(f"[Rank {local_rank}] Resumed from epoch {start_epoch}")

    # -------------------- Training Loop --------------------
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Rank {local_rank}]") if local_rank == 0 else dataloader

        for batch in pbar:
            step = batch['step'].to(device)
            bar = batch['bar'].to(device)
            phrase = batch['phrase'].to(device)
            metrics_list = batch['metrics']

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                step_pred, bar_pred, phrase_pred = model(step, bar, phrase)
                loss_step = bce_loss(step_pred, step)
                loss_bar = bce_loss(bar_pred, bar) if bar_pred is not None else 0.0
                loss_phrase = bce_loss(phrase_pred, phrase) if phrase_pred is not None else 0.0
                supervised_loss = loss_step + loss_bar + loss_phrase

                rl_loss = 0.0
                adaptive_weights = None
                if args.rl_enabled:
                    reward, adaptive_weights = compute_reward(step_pred, metrics_list[0], device)
                    rl_loss = -args.rl_weight * reward

                total_loss = supervised_loss + rl_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()

        scheduler.step()
        if local_rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            msg = f"Epoch {epoch}: loss={avg_loss:.6f}"
            if adaptive_weights is not None:
                msg += f", adaptive_weights={adaptive_weights.cpu().numpy()}"
            print(msg)
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))

    if world_size > 1:
        dist.destroy_process_group()
        print(f"[Rank {local_rank}] DDP process group destroyed")

# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--rl_enabled', action='store_true')
    parser.add_argument('--rl_weight', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_drums', type=int, default=9)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    world_size = args.num_gpus

    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(0, world_size, args)
