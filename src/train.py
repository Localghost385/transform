import os
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import CompleteHierarchicalDrumDataset
from model import HierarchicalDrumModel

# ---------------------------
# Collate (unchanged semantics)
# ---------------------------
def collate_fn(batch):
    return {
        "step": nn.utils.rnn.pad_sequence([d["step"] for d in batch], batch_first=True),
        "bar": nn.utils.rnn.pad_sequence([d["bar"] for d in batch], batch_first=True),
        "phrase": nn.utils.rnn.pad_sequence([d["phrase"] for d in batch], batch_first=True),
        "metrics": [d["metrics"] for d in batch]
    }

# ---------------------------
# Batched RL reward (vectorized)
# ---------------------------
def compute_reward_batch(step_probs, metrics_list, device):
    """
    step_probs: (B, T, D) tensor (after sigmoid) - unused directly in this reward impl,
                 but kept as an argument if you want to expand reward using probs later.
    metrics_list: list of length B of metric dicts
    Returns:
        rewards: (B,)
        weights: (B, 3)
    """
    dense_fracs = []
    syncs = []
    ioi_vars = []

    for m in metrics_list:
        m_step = m["step"]
        dense_fracs.append(m_step["dense_frac"])
        syncs.append(m_step["sync"])
        # ioi_stats expected shape (N_intervals, stats) where index 1 is variance or similar
        ioi_vars.append(m_step["ioi_stats"][:, 1].mean())

    dense_fracs = torch.tensor(dense_fracs, dtype=torch.float32, device=device)
    syncs = torch.tensor(syncs, dtype=torch.float32, device=device)
    ioi_vars = torch.tensor(ioi_vars, dtype=torch.float32, device=device)

    weights = torch.stack([dense_fracs, syncs, ioi_vars], dim=1)  # (B,3)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    features = torch.stack([dense_fracs, syncs, 1.0 / (ioi_vars + 1e-6)], dim=1)  # (B,3)

    rewards = (weights * features).sum(dim=1)  # (B,)
    return rewards, weights

# ---------------------------
# Checkpoint utils
# ---------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint["epoch"]

# ---------------------------
# Training loop (optimized for 2x T4)
# ---------------------------
def train(rank, world_size, local_rank, args):
    # set device and seeds
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    seed = 42 + rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # faster cuDNN for fixed-size-ish operations
    torch.backends.cudnn.benchmark = True

    # init DDP
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"[rank {rank}] init process group, local_rank={local_rank}")

    # dataset + dataloader
    dataset = CompleteHierarchicalDrumDataset(args.data_dir, seq_len=args.seq_len, augment=args.augment)
    sampler = DistributedSampler(dataset) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=min(8, max(2, args.prefetch_factor)),
        drop_last=True  # helps even workload across GPUs
    )

    # determine model dims
    sample = dataset[0]
    num_drums = sample["step"].shape[-1]

    # create model and move to device (after set_device)
    model = HierarchicalDrumModel(
        num_drums=num_drums,
        step_hidden_dim=args.d_model,
        bar_hidden_dim=args.d_model,
        phrase_hidden_dim=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # DDP wrap
    ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    # allow overriding if user needs find_unused
    if args.find_unused_parameters:
        ddp_kwargs["find_unused_parameters"] = True

    if world_size > 1:
        model = DDP(model, **ddp_kwargs)
        print(f"[rank {rank}] model wrapped in DDP")

    # optimizer, scheduler, amp
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # loss
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # resume if requested
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        print(f"[rank {rank}] resumed from epoch {start_epoch}")

    is_main = (rank == 0)

    # training loop
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main)

        t0 = time.time()
        for batch in pbar:
            # non-blocking transfers (requires pin_memory=True)
            step = batch["step"].to(device, non_blocking=True)
            bar = batch["bar"].to(device, non_blocking=True)
            phrase = batch["phrase"].to(device, non_blocking=True)
            metrics_list = batch["metrics"]  # still a python list of dicts; compute_reward_batch handles vectorization

            optimizer.zero_grad()

            # autocast for T4 (fp16) is a good fit
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                step_logits, bar_logits, phrase_logits = model(step, bar, phrase)

                # BCE on whole padded sequences — keep semantics same as original
                loss_step = bce_loss(step_logits, step)
                loss_bar = bce_loss(bar_logits, bar) if bar_logits is not None else 0.0*model.bar_layer.weight.sum()
                loss_phrase = bce_loss(phrase_logits, phrase) if phrase_logits is not None else 0.0*model.phrase_layer.weight.sum()

                supervised_loss = loss_step + loss_bar + loss_phrase

                rl_loss = 0.0
                adaptive_weights = None

                if args.rl_enabled:
                    # compute step probabilities (not used in the reward now but included for future extensions)
                    step_probs = torch.sigmoid(step_logits)
                    # compute reward on GPU without affecting grad graph
                    with torch.no_grad():
                        rewards, adaptive_weights = compute_reward_batch(step_probs, metrics_list, device)
                    # aggregate reward into scalar loss. Using mean is stable.
                    rl_loss = -args.rl_weight * rewards.mean()

                total_loss = supervised_loss + rl_loss

            # backward with amp scaler
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                # unscale before clipping per AMP best practice
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1

            if is_main:
                pbar.set_postfix(loss=epoch_loss / batch_count)

        # scheduler step at epoch end
        scheduler.step()

        # logging & checkpointing by main process only
        if is_main:
            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(1, batch_count)
            msg = f"Epoch {epoch}: loss={avg_loss:.6f}, time={elapsed:.1f}s"
            if adaptive_weights is not None:
                msg += f", adaptive_weights_mean={adaptive_weights.mean(dim=0).cpu().numpy()}"
            print(msg)

            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            )

    # cleanup
    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="per-process (per-GPU) batch size")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--rl_enabled", action="store_true")
    parser.add_argument("--rl_weight", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--find_unused_parameters", action="store_true",
                        help="set if your model legitimately has unused params (slower).")
    return parser.parse_args()

# ---------------------------
# Main (torchrun entrypoint)
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # torchrun provides these
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    train(rank, world_size, local_rank, args)
