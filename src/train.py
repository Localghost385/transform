import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import CachedHierarchicalDrumDataset  # <-- updated import
from model import HierarchicalDrumModel

# -------------------- Collate Function --------------------
def collate_fn(batch):
    # Cached dataset already returns torch tensors
    return {
        "step": torch.stack([d["step"] for d in batch]),
        "bar": torch.stack([d["bar"] for d in batch]),
        "phrase": torch.stack([d["phrase"] for d in batch]),
        "metrics": [d["metrics"] for d in batch]
    }

# -------------------- Batched RL Reward --------------------
def compute_hierarchical_reward(step_probs, bar_probs, phrase_probs, metrics_list, device):
    """Compute hierarchical RL rewards per level and adaptive weights."""
    
    def _compute_level_rewards(probs, metrics, feature_keys):
        dense_fracs, syncs, ioi_vars = [], [], []
        for m in metrics:
            dense_fracs.append(m['dense_frac'])
            syncs.append(m['sync'])
            ioi_vars.append(m['ioi_stats'][:, 1].mean())
        dense_fracs = torch.tensor(dense_fracs, device=device, dtype=torch.float32)
        syncs = torch.tensor(syncs, device=device, dtype=torch.float32)
        ioi_vars = torch.tensor(ioi_vars, device=device, dtype=torch.float32)
        features = torch.stack([
            dense_fracs / (dense_fracs.max() + 1e-6),
            syncs / (syncs.max() + 1e-6),
            1.0 / (ioi_vars + 1e-6)
        ], dim=1)
        features = features / (features.sum(dim=1, keepdim=True) + 1e-6)
        adaptive_weights = torch.softmax(features, dim=1)
        rewards = (adaptive_weights * features).sum(dim=1)
        return rewards, adaptive_weights

    step_rewards, step_weights = _compute_level_rewards(step_probs, [m['step'] for m in metrics_list], ['dense_frac','sync','ioi'])
    bar_rewards, bar_weights = _compute_level_rewards(bar_probs, [m['bar'] for m in metrics_list], ['dense_frac','sync','ioi'])
    phrase_rewards, phrase_weights = _compute_level_rewards(phrase_probs, [m['phrase'] for m in metrics_list], ['dense_frac','sync','ioi'])

    # Combine hierarchical rewards (you can tune these weights)
    rl_rewards = 0.5*step_rewards + 0.3*bar_rewards + 0.2*phrase_rewards
    return rl_rewards, (step_weights, bar_weights, phrase_weights)

    dense_fracs = []
    syncs = []
    ioi_vars = []

    for m in metrics_list:
        m_step = m['step']
        dense_fracs.append(m_step['dense_frac'])
        syncs.append(m_step['sync'])
        ioi_vars.append(m_step['ioi_stats'][:, 1].mean())

    dense_fracs = torch.tensor(dense_fracs, device=device, dtype=torch.float32)
    syncs = torch.tensor(syncs, device=device, dtype=torch.float32)
    ioi_vars = torch.tensor(ioi_vars, device=device, dtype=torch.float32)

    # Feature normalization (min-max or scale to [0,1])
    features = torch.stack([
        dense_fracs / (dense_fracs.max() + 1e-6),
        syncs / (syncs.max() + 1e-6),
        1.0 / (ioi_vars + 1e-6)
    ], dim=1)
    features = features / (features.sum(dim=1, keepdim=True) + 1e-6)

    # Softmax to compute adaptive weights
    adaptive_weights = torch.softmax(features, dim=1)

    # Reward = weighted sum
    rewards = (adaptive_weights * features).sum(dim=1)
    return rewards, adaptive_weights


# -------------------- Checkpointing --------------------
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint['epoch']


# -------------------- Training Loop --------------------
def train(rank, world_size, args):
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # DDP initialization
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            rank=local_rank,
            world_size=world_size
        )
        print(f"[Rank {local_rank}] DDP initialized with {world_size} GPUs")

    # -------------------- Dataset & DataLoader --------------------
    dataset = CachedHierarchicalDrumDataset(
        args.data_dir,
        seq_len=args.seq_len,
        augment=args.augment
    )
    sampler = DistributedSampler(dataset) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Get number of drums dynamically
    sample = dataset[0]
    num_drums = sample['step'].shape[-1]

    # -------------------- Model --------------------
    model = HierarchicalDrumModel(
        num_drums=num_drums,
        step_hidden_dim=args.d_model,
        bar_hidden_dim=args.d_model,
        phrase_hidden_dim=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        print(f"[Rank {local_rank}] Model wrapped in DDP")

    # -------------------- Optimizer & Scheduler --------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    bce_loss = nn.BCEWithLogitsLoss()
    start_epoch = 0

    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        print(f"[Rank {local_rank}] Resumed from epoch {start_epoch}")

    # -------------------- Training Loop --------------------
    for epoch in range(start_epoch, args.epochs):
        if sampler:
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

            with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                step_logits, bar_logits, phrase_logits = model(step, bar, phrase)

                loss_step = bce_loss(step_logits, step)
                loss_bar = bce_loss(bar_logits, bar) if bar_logits is not None else 0.0
                loss_phrase = bce_loss(phrase_logits, phrase) if phrase_logits is not None else 0.0

                supervised_loss = loss_step + loss_bar + loss_phrase

                rl_loss = 0.0
                adaptive_weights = None

                if args.rl_enabled:
                    step_probs = torch.sigmoid(step_logits)
                    bar_probs = torch.sigmoid(bar_logits) if bar_logits is not None else None
                    phrase_probs = torch.sigmoid(phrase_logits) if phrase_logits is not None else None
                    with torch.no_grad():
                        rewards, adaptive_weights = compute_hierarchical_reward(step_probs, bar_probs, phrase_probs, metrics_list, device)
                    rl_loss = -args.rl_weight * rewards.mean()

                total_loss = supervised_loss + rl_loss

            # -------------------- Backprop --------------------
            if scaler:
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            epoch_loss += total_loss.item()

        scheduler.step()

        if local_rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            msg = f"Epoch {epoch}: loss={avg_loss:.6f}"
            if adaptive_weights is not None:
                msg += f", adaptive_weights_mean={adaptive_weights.mean(dim=0).cpu().numpy()}"
            print(msg)

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            )

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
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)  # Cached dataset is memory-heavy
    return parser.parse_args()


# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    world_size = args.num_gpus
    train(0, world_size, args)
