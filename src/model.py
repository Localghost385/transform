import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------- Hierarchical Drum Model --------------------
class HierarchicalDrumModel(nn.Module):
    """
    Hierarchical drum model supporting step, bar, and phrase levels.
    Can use RNNs (default GRU) or Transformer encoders.
    """

    def __init__(
        self,
        num_drums: int,
        step_hidden_dim: int = 128,
        bar_hidden_dim: int = 128,
        phrase_hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_transformer: bool = False,
    ):
        super().__init__()
        self.num_drums = num_drums
        self.step_hidden_dim = step_hidden_dim
        self.bar_hidden_dim = bar_hidden_dim
        self.phrase_hidden_dim = phrase_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_transformer = use_transformer

        # Step-level encoder
        if use_transformer:
            self.step_rnn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=step_hidden_dim, nhead=4, dim_feedforward=step_hidden_dim*4, dropout=dropout
                ),
                num_layers=num_layers
            )
        else:
            self.step_rnn = nn.GRU(num_drums, step_hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Linear projection from step to bar
        self.step_to_bar = nn.Linear(step_hidden_dim, bar_hidden_dim)

        # Bar-level encoder
        if use_transformer:
            self.bar_rnn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=bar_hidden_dim, nhead=4, dim_feedforward=bar_hidden_dim*4, dropout=dropout
                ),
                num_layers=num_layers
            )
        else:
            self.bar_rnn = nn.GRU(bar_hidden_dim, bar_hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Linear projection from bar to phrase
        self.bar_to_phrase = nn.Linear(bar_hidden_dim, phrase_hidden_dim)

        # Phrase-level encoder
        if use_transformer:
            self.phrase_rnn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=phrase_hidden_dim, nhead=4, dim_feedforward=phrase_hidden_dim*4, dropout=dropout
                ),
                num_layers=num_layers
            )
        else:
            self.phrase_rnn = nn.GRU(phrase_hidden_dim, phrase_hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # Final step output layer
        self.output_layer = nn.Linear(step_hidden_dim + bar_hidden_dim + phrase_hidden_dim, num_drums)
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(dropout)

    # -------------------- Forward --------------------
    def forward(
        self,
        step_seq: torch.Tensor,
        bar_seq: Optional[torch.Tensor] = None,
        phrase_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        step_seq: [batch, seq_len, num_drums]
        bar_seq: [batch, num_bars, num_drums] optional (used if aggregation is skipped)
        phrase_seq: [batch, num_phrases, num_drums] optional
        Returns:
            step_preds: [batch, seq_len, num_drums]
            bar_preds: [batch, num_bars, num_drums] (optional)
            phrase_preds: [batch, num_phrases, num_drums] (optional)
        """
        batch_size, seq_len, D = step_seq.shape

        # Step-level encoding
        if self.use_transformer:
            # Transformer expects [seq_len, batch, d_model], project first
            step_embed = F.linear(step_seq, torch.eye(D, self.step_hidden_dim, device=step_seq.device))
            step_out = self.step_rnn(step_embed.transpose(0,1)).transpose(0,1)
        else:
            step_out, _ = self.step_rnn(step_seq)  # [batch, seq_len, step_hidden_dim]

        # Aggregate step -> bar
        num_bars = step_out.shape[1] // 16
        step_trim = step_out[:, :num_bars*16, :]  # trim to full bars
        step_bar = step_trim.reshape(batch_size, num_bars, 16, self.step_hidden_dim).mean(dim=2)
        bar_in = self.step_to_bar(step_bar)
        # Bar-level encoding
        if self.use_transformer:
            bar_out = self.bar_rnn(bar_in.transpose(0,1)).transpose(0,1)
        else:
            bar_out, _ = self.bar_rnn(bar_in)

        # Aggregate bar -> phrase
        num_phrases = bar_out.shape[1] // 4
        bar_trim = bar_out[:, :num_phrases*4, :]
        bar_phrase = bar_trim.reshape(batch_size, num_phrases, 4, self.bar_hidden_dim).mean(dim=2)
        phrase_in = self.bar_to_phrase(bar_phrase)
        # Phrase-level encoding
        if self.use_transformer:
            phrase_out = self.phrase_rnn(phrase_in.transpose(0,1)).transpose(0,1)
        else:
            phrase_out, _ = self.phrase_rnn(phrase_in)

        # Broadcast bar and phrase embeddings back to step-level
        bar_broadcast = bar_out.repeat_interleave(16, dim=1)
        phrase_broadcast = phrase_out.repeat_interleave(4*16, dim=1)  # 4 bars per phrase

        # Trim to match seq_len
        bar_broadcast = bar_broadcast[:, :seq_len, :]
        phrase_broadcast = phrase_broadcast[:, :seq_len, :]

        # Concatenate embeddings and predict
        combined = torch.cat([step_out, bar_broadcast, phrase_broadcast], dim=-1)
        combined = self.dropout_layer(combined)
        step_preds = self.output_layer(combined)  # logits
        step_probs = self.activation(step_preds)

        # Optional outputs at higher levels
        bar_preds = self.activation(bar_out @ self.output_layer.weight.T)
        phrase_preds = self.activation(phrase_out @ self.output_layer.weight.T)

        return step_probs, bar_preds, phrase_preds

    # -------------------- Loss --------------------
    def compute_loss(
        self,
        step_preds: torch.Tensor,
        step_targets: torch.Tensor,
        bar_preds: Optional[torch.Tensor] = None,
        bar_targets: Optional[torch.Tensor] = None,
        phrase_preds: Optional[torch.Tensor] = None,
        phrase_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hierarchical BCE loss
        """
        loss = F.binary_cross_entropy(step_preds, step_targets)
        if bar_preds is not None and bar_targets is not None:
            loss += F.binary_cross_entropy(bar_preds, bar_targets)
        if phrase_preds is not None and phrase_targets is not None:
            loss += F.binary_cross_entropy(phrase_preds, phrase_targets)
        return loss

    # -------------------- Autoregressive generation --------------------
    @torch.no_grad()
    def generate(
        self,
        start_seq: torch.Tensor,
        length: int = 512,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive generation (step-level)
        """
        generated = [start_seq]
        current_seq = start_seq.clone()
        for _ in range(length):
            step_pred, _, _ = self.forward(current_seq)
            last_step = step_pred[:, -1, :]  # take last predicted step
            prob = last_step / temperature
            next_step = torch.bernoulli(prob)
            generated.append(next_step)
            current_seq = torch.cat([current_seq, next_step.unsqueeze(1)], dim=1)
        return torch.cat(generated, dim=1)

