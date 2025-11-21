import os
import time
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataset import DrumDataset
from model import DrumTransformer
from tqdm import tqdm
from contextlib import nullcontext

# -------------------------
# Helper utilities
# -------------------------
def setup_device_and_perf(args):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"[DEBUG] Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        print(f"[DEBUG] CUDA available devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[DEBUG] Device {i} name: {torch.cuda.get_device_name(i)}")
            print(f"[DEBUG] Device {i} memory allocated: {torch.cuda.memory_allocated(i)/1e6:.2f} MB")
            print(f"[DEBUG] Device {i} memory reserved: {torch.cuda.memory_reserved(i)/1e6:.2f} MB")
    else:
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(args.num_threads)
            torch.set_num_interop_threads(args.num_threads)
        print(f"[DEBUG] CPU threads set to: {args.num_threads}")
    return device

def build_optimizer(model, lr, weight_decay=0.01):
    print(f"[DEBUG] Building optimizer with lr={lr}, weight_decay={weight_decay}")
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8)

def build_scheduler_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = float(max(0, step - warmup_steps)) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    print(f"[DEBUG] Scheduler built with warmup_steps={warmup_steps}, total_steps={total_steps}")
    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, is_best=False):
    print(f"[DEBUG] Saving checkpoint at {path}")
    if hasattr(model, "module"):  # unwrap DDP
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "time": time.time(),
    }
    torch.save(ckpt, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pt")
        torch.save(ckpt, best_path)
        print(f"[DEBUG] Saved best model to {best_path}")

def maybe_load_checkpoint(resume_path, model, optimizer=None, scaler=None, scheduler=None, map_location="cpu"):
    if resume_path is None or not os.path.exists(resume_path):
        print("[DEBUG] No checkpoint found to resume")
        return 0, 0, False
    print(f"[DEBUG] Loading checkpoint from {resume_path}")
    ckpt = torch.load(resume_path, map_location=map_location)
    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"[DEBUG] Checkpoint loaded: start_epoch={start_epoch}, global_step={global_step}")
    return start_epoch + 1, global_step, True

# -------------------------
# Training / Eval loops
# -------------------------
def train_epoch(model, dataloader, optimizer, scaler, device, scheduler=None,
                clip_grad=1.0, grad_accum=1, use_amp=True, prog_desc="Training", start_step=0):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    global_step = start_step

    autocast_ctx = nullcontext
    if use_amp:
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)

    optimizer.zero_grad()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=prog_desc) if dist.get_rank() == 0 else enumerate(dataloader)
    start_time = time.time()
    for batch_idx, (x, y) in pbar:
        batch_start = time.time()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx():
            preds = model(x)
            loss = model.compute_loss(preds, y) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        batch_n = x.size(0)
        total_loss += loss.item() * batch_n * grad_accum
        total_tokens += batch_n
        avg_loss = total_loss / max(1, total_tokens)
        if dist.get_rank() == 0:
            pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}", "step": global_step})
            print(f"[DEBUG] Batch {batch_idx} processed in {time.time() - batch_start:.2f}s, GPU mem allocated: {torch.cuda.memory_allocated()/1e6:.2f}MB")

    print(f"[DEBUG] Epoch finished in {time.time() - start_time:.2f}s")
    return avg_loss, global_step

def eval_epoch(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    autocast_ctx = nullcontext
    if use_amp:
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", total=len(dataloader)) if dist.get_rank() == 0 else dataloader
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_ctx():
                preds = model(x)
                loss = model.compute_loss(preds, y)
            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_tokens += batch_n
            avg_loss = total_loss / max(1, total_tokens)
            if dist.get_rank() == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
    return avg_loss

# -------------------------
# Main training entry
# -------------------------
def main(args):
    num_gpus = torch.cuda.device_count()
    print(f"[DEBUG] Detected {num_gpus} GPUs.")

    device = setup_device_and_perf(args)
    print(f"[DEBUG] Using device: {device}")

    # ---- datasets ----
    t0 = time.time()
    train_dataset = DrumDataset(args.train_dir, seq_len=args.seq_len, augment=True)
    val_dataset = DrumDataset(args.val_dir, seq_len=args.seq_len, augment=False)
    print(f"[DEBUG] Datasets loaded in {time.time() - t0:.2f}s: train={len(train_dataset)}, val={len(val_dataset)}")

    # ---- samplers / dataloaders ----
    if num_gpus > 1:
        if not hasattr(args, "local_rank"):
            raise ValueError("local_rank missing! Pass --local_rank when running torchrun")
        dist.init_process_group(backend="nccl", rank=args.local_rank, world_size=num_gpus)
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=max(0, args.num_workers // 2), pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers // 2), pin_memory=True)
        train_sampler = None

    print(f"[DEBUG] DataLoaders ready, first batch sample shapes: {train_dataset[0][0].shape}, {train_dataset[0][1].shape}")

    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / args.batch_size / max(1, args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    print(f"[DEBUG] Total steps calculated: {total_steps} ({steps_per_epoch} per epoch)")

    # ---- model ----
    model = DrumTransformer(
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout
    ).to(device)
    print(f"[DEBUG] Model instantiated: {sum(p.numel() for p in model.parameters()):,} parameters")

    if getattr(args, "use_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[DEBUG] Model compiled with torch.compile()")
        except Exception as e:
            print("[DEBUG] torch.compile failed:", e)

    if num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        print("[DEBUG] Model wrapped in DDP")

    # ---- optimizer / scaler / scheduler ----
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.01))
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.use_amp) else None
    warmup_steps = int(total_steps * getattr(args, "warmup_ratio", 0.03)) if getattr(args, "warmup_steps", None) is None else int(getattr(args, "warmup_steps", 100))
    scheduler = build_scheduler_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=max(1, total_steps))

    # ---- resume support ----
    start_epoch = 1
    global_step = 0
    if getattr(args, "resume_from", None):
        s_epoch, gstep, loaded = maybe_load_checkpoint(args.resume_from, model, optimizer=optimizer, scaler=scaler, scheduler=scheduler, map_location=device)
        if loaded:
            start_epoch = s_epoch
            global_step = gstep
            print(f"[DEBUG] Resuming from {args.resume_from}: start_epoch={start_epoch}, global_step={global_step}")

    # ---- training loop ----
    best_val = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if num_gpus > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print(f"[DEBUG] Starting epoch {epoch}/{args.epochs}")

        train_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            scheduler=scheduler,
            clip_grad=args.clip_grad,
            grad_accum=args.grad_accum,
            use_amp=args.use_amp,
            prog_desc=f"Train E{epoch}",
            start_step=global_step
        )

        print(f"[DEBUG] Epoch {epoch} train loss: {train_loss:.6f} (global_step={global_step})")

        # Checkpoint
        if args.save_every_steps and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
            step_ckpt = os.path.join(args.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
            save_checkpoint(step_ckpt, model, optimizer, scaler, scheduler, epoch, global_step)

        # Evaluate
        val_loss = eval_epoch(model, val_loader, device, use_amp=args.use_amp)
        print(f"[DEBUG] Epoch {epoch} validation loss: {val_loss:.6f}")

        # Save epoch checkpoint
        epoch_ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_checkpoint(epoch_ckpt, model, optimizer, scaler, scheduler, epoch, global_step)

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pt")
            save_checkpoint(best_ckpt, model, optimizer, scaler, scheduler, epoch, global_step)
            print(f"[DEBUG] New best model saved at {best_ckpt}")

    if num_gpus > 1:
        dist.destroy_process_group()

    print("[DEBUG] Training complete.")
