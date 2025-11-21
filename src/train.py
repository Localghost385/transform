import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import DrumDataset, get_dataloader
from model import DrumTransformer
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = model.compute_loss(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = model.compute_loss(preds, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

def main(args):
    """
    Train Drum Transformer using args from CLI.
    """
    # Data loaders
    train_loader = get_dataloader(args.train_dir, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.val_dir, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False)

    # Model
    model = DrumTransformer(
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout
    ).to(args.device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, args.device, clip_grad=args.clip_grad)
        print(f"Train Loss: {train_loss:.6f}")

        val_loss = eval_epoch(model, val_loader, args.device)
        print(f"Validation Loss: {val_loss:.6f}")

        scheduler.step()

        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            model.save(best_path)
            print(f"Saved best model to {best_path}")
