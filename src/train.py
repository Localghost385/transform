import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataset import get_dataloader
from model import DrumTransformer
from tqdm import tqdm

# -------------------------
# Helper utilities
# -------------------------
def setup_device_and_perf(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # CPU / GPU tuning
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # allow TF32 if available (works on Ampere+)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    else:
        # Improve CPU throughput
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(args.num_threads)
            torch.set_num_interop_threads(args.num_threads)

    return device

def build_optimizer(model, lr, weight_decay=0.01):
    # pick sensible defaults for transformers
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8)

def build_scheduler_with_warmup(optimizer, warmup_steps, total_steps):
    """
    Return a LambdaLR that does linear warmup then cosine decay to zero.
    """
    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay after warmup
        progress = float(max(0, step - warmup_steps)) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, step, is_best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "time": time.time(),
    }
    torch.save(ckpt, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pt")
        torch.save(ckpt, best_path)

def maybe_load_checkpoint(resume_path, model, optimizer=None, scaler=None, scheduler=None, map_location="cpu"):
    """
    Load checkpoint if exists. Returns (start_epoch, global_step, loaded_flag)
    """
    if resume_path is None or not os.path.exists(resume_path):
        return 0, 0, False
    ckpt = torch.load(resume_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return start_epoch + 1, global_step, True

# -------------------------
# Training / Eval loops
# -------------------------
def train_epoch(
    model, dataloader, optimizer, scaler, device, scheduler=None,
    clip_grad=1.0, grad_accum=1, use_amp=True, prog_desc="Training", start_step=0
):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    global_step = start_step

    # choose autocast dtype for CPU/GPU
    from contextlib import nullcontext

    if use_amp:
        if device.type == "cuda":
            autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.float16)
        else:
            autocast_ctx = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext  # no-op context manager


    optimizer.zero_grad()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=prog_desc)
    for batch_idx, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)
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

        # bookkeeping
        batch_n = x.size(0)
        total_loss += loss.item() * batch_n * grad_accum  # loss was scaled down earlier
        total_tokens += batch_n
        avg_loss = total_loss / max(1, total_tokens)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}", "step": global_step})

    return avg_loss, global_step

def eval_epoch(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        # use same autocast behavior as training for consistent perf
        from contextlib import nullcontext

        if use_amp:
            if device.type == "cuda":
                autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.float16)
            else:
                autocast_ctx = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext  # no-op context manager

        pbar = tqdm(dataloader, desc="Validation", total=len(dataloader))
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            with autocast_ctx():
                preds = model(x)
                loss = model.compute_loss(preds, y)
            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_tokens += batch_n
            avg_loss = total_loss / max(1, total_tokens)
            pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
    return avg_loss

# -------------------------
# Main training entry (args-driven)
# -------------------------
def main(args):
    """
    args should provide:
      - train_dir, val_dir
      - num_classes, seq_len, d_model, nhead, num_layers, ff_dim, dropout
      - batch_size, epochs, lr, clip_grad, checkpoint_dir
      - device (e.g., "cuda" or "cpu"), num_workers, num_threads
      - grad_accum, warmup_steps (or warmup_ratio), use_amp (bool), use_compile (bool)
      - save_every_steps (int), resume_from (path or None)
    """
    # ---- perf / device setup ----
    device = setup_device_and_perf(args)
    print("Using device:", device)

    # ---- dataloaders ----
    dl_kwargs = dict(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        augment=False,
        num_workers=args.num_workers
    )
    # get_dataloader constructs dataset internally; ensure persistent_workers on PyTorch >=1.7
    train_loader = get_dataloader(args.train_dir, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True, augment=False, num_workers=args.num_workers)
    val_loader = get_dataloader(args.val_dir, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False, augment=False, num_workers=max(0, args.num_workers//2))

    steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / args.batch_size / max(1, args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    print(f"Dataset samples: {len(train_loader.dataset)}; steps/epoch (effective): {steps_per_epoch}; total_steps: {total_steps}")

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

    if getattr(args, "use_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print("torch.compile failed:", e)

    print(f"Model parameters: {model.count_parameters():,}")

    # ---- optimizer / scaler / scheduler ----
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay if hasattr(args, "weight_decay") else 0.01)

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.use_amp) else (
        torch.cuda.amp.GradScaler() if (device.type == "cpu" and args.use_amp and getattr(torch.cuda.amp, "GradScaler", None) is not None) else None
    )

    # Compute warmup steps (allow warmup_ratio)
    if getattr(args, "warmup_steps", None) is None and getattr(args, "warmup_ratio", None) is not None:
        warmup_steps = int(total_steps * args.warmup_ratio)
    else:
        warmup_steps = int(getattr(args, "warmup_steps", 100))

    scheduler = build_scheduler_with_warmup(optimizer, warmup_steps=warmup_steps, total_steps=max(1, total_steps))

    # ---- resume support ----
    start_epoch = 1
    global_step = 0
    if getattr(args, "resume_from", None):
        s_epoch, gstep, loaded = maybe_load_checkpoint(args.resume_from, model, optimizer=optimizer, scaler=scaler, scheduler=scheduler, map_location=device)
        if loaded:
            start_epoch = s_epoch
            global_step = gstep
            print(f"Resuming from {args.resume_from}: start_epoch={start_epoch}, global_step={global_step}")

    # ---- training loop ----
    best_val = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Train
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

        print(f"Epoch {epoch} train loss: {train_loss:.6f} (global_step={global_step})")

        # Periodic checkpointing by step
        if args.save_every_steps and args.save_every_steps > 0:
            if global_step % args.save_every_steps == 0:
                step_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                save_checkpoint(step_checkpoint, model, optimizer, scaler, scheduler, epoch, global_step)
                print("Saved step checkpoint:", step_checkpoint)

        # Evaluate
        val_loss = eval_epoch(model, val_loader, device, use_amp=args.use_amp)
        print(f"Epoch {epoch} validation loss: {val_loss:.6f}")

        # Scheduler already stepped each optimizer.step() if scheduler provided and used that way;
        # If you prefer per-epoch stepping, uncomment:
        # scheduler.step()

        # Save epoch checkpoint and best model
        epoch_ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_checkpoint(epoch_ckpt, model, optimizer, scaler, scheduler, epoch, global_step)
        print("Saved epoch checkpoint:", epoch_ckpt)

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pt")
            save_checkpoint(best_ckpt, model, optimizer, scaler, scheduler, epoch, global_step)
            print("Saved new best model:", best_ckpt)

    print("Training complete.")
