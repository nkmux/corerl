import torch
import torch.nn as nn
import math

from .definitions import TARGET_NAMES

INDOOR_T_IDX = TARGET_NAMES.index("indoor_dry_bulb_temperature")

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x):  # x: [B,T,D]
        return x + self.pe[:x.size(1)].unsqueeze(0)

class TransformerDynamicsModel(nn.Module):
    def __init__(self, input_dim, target_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, target_dim)

    def forward(self, x):                    # x: [B, T, F]
        x = self.input_proj(x)               # [B, T, D]
        x = self.pos_enc(x)                  # add sinusoidal PE
        h = self.encoder(x)                  # [B, T, D]
        y = h[:, -1, :]                      # last step
        return self.output_proj(y)           # [B, target_dim]
