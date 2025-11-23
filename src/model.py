import torch
import torch.nn as nn
import torch.nn.functional as F

class DrumTransformer(nn.Module):
    """
    Decoder-only Transformer (GPT-style) for drum pattern generation.
    Input: multi-hot vectors of drum hits (T, D)
    Output: probabilities for each drum class (T, D)
    """
    def __init__(
        self,
        num_classes: int = 23,      # D
        seq_len: int = 512,         # context window
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.d_model = d_model

        # Input embedding: project multi-hot drum vector -> d_model
        self.input_proj = nn.Linear(num_classes, d_model)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, D) -> multi-hot drum sequence
        Returns:
            out: Tensor of shape (B, T, D) -> probabilities for each drum class
        """
        B, T, D = x.shape
        device = x.device
        assert D == self.num_classes, f"Input D={D}, expected {self.num_classes}"

        # Input projection
        x = self.input_proj(x) * (self.d_model ** 0.5)  # (B, T, d_model)

        # Positional embeddings
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embed(pos_ids)

        # Transformer expects (T, B, d_model)
        x = x.transpose(0, 1)  # (T, B, d_model)

        # Causal mask for autoregressive prediction
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)  # (T, T)

        # Pass through transformer decoder
        out = self.transformer(x, x, tgt_mask=mask)  # (T, B, d_model)

        # Back to (B, T, d_model)
        out = out.transpose(0, 1)

        # Output projection + sigmoid for multi-label
        out = torch.sigmoid(self.output_proj(out))  # (B, T, D)
        return out

    def compute_loss(self, preds, targets):
        """
        Binary Cross-Entropy loss over all drum classes per time step.
        Args:
            preds: (B, T, D) - probabilities
            targets: (B, T, D) - multi-hot ground truth
        Returns:
            loss: scalar
        """
        return F.binary_cross_entropy(preds, targets)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None, strict=False, print_missing=True):
        """
        Robustly load a checkpoint or bare state_dict.

        - If path points to a training checkpoint (dict with "model_state_dict"), that is extracted.
        - Removes "module." prefixes (from DataParallel / DDP) automatically.
        - Calls load_state_dict(..., strict=strict).
        - If print_missing=True, prints missing/unexpected keys summary.
        """
        ckpt = torch.load(path, map_location=map_location)

        # If it's a training checkpoint, extract model_state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # probably already a raw state_dict
            state_dict = ckpt
        else:
            state_dict = ckpt

        # Remove 'module.' prefix if present (from DataParallel / DDP)
        new_state = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("module."):
                new_k = k[len("module."):]
            new_state[new_k] = v

        # Try to load
        missing_keys, unexpected_keys = self.load_state_dict(new_state, strict=strict)

        # For modern torch versions load_state_dict returns NamedTuple with missing/unexpected.
        # If it returned None (older style) we can attempt to deduce nothing further here.
        try:
            missing = missing_keys.missing_keys
            unexpected = missing_keys.unexpected_keys
        except Exception:
            # Fallback for older torch - ignore
            missing = None
            unexpected = None

        if print_missing:
            if missing is not None:
                if len(missing) > 0:
                    print("[LOAD] Missing keys:", missing)
                else:
                    print("[LOAD] No missing keys.")
            if unexpected is not None:
                if len(unexpected) > 0:
                    print("[LOAD] Unexpected keys:", unexpected)
                else:
                    print("[LOAD] No unexpected keys.")

        return missing, unexpected



if __name__ == "__main__":
    # Quick sanity check
    model = DrumTransformer(num_classes=23, seq_len=512)
    x = torch.randint(0, 2, (2, 512, 23)).float()  # batch_size=2
    out = model(x)
    print("Output shape:", out.shape)
    print("Number of trainable parameters:", model.count_parameters())
