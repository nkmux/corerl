import torch
import torch.nn as nn
from collections import deque
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

class TransformerDynamicsAdapter:
    """CityLearn-compatible dynamics: called as (x_window, hidden_state) -> (y_norm, h_next)."""
    def __init__(self, filepath: str):
        ckpt = torch.load(filepath, map_location="cpu")

        # Rebuild model
        arch = ckpt["arch"]
        self.model = TransformerDynamicsModel(**arch)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # IO config
        self.input_observation_names  = ckpt["input_observation_names"]
        self.lookback     = int(ckpt["lookback"])
        self.n_features   = len(self.input_observation_names)

        self.input_normalization_minimum = torch.tensor(
            ckpt["input_normalization_minimum"], dtype=torch.float32
        )
        self.input_normalization_maximum = torch.tensor(
            ckpt["input_normalization_maximum"], dtype=torch.float32
        )
        
        # Target scaling (if present in your training)
        self.target_names = ckpt.get("target_names", None)
        if self.target_names is None:
            self.target_names = TARGET_NAMES

        self.indoor_idx = self.target_names.index("indoor_dry_bulb_temperature")

        self.has_y_scale = ("target_minimum" in ckpt and "target_maximum" in ckpt)
        if self.has_y_scale:
            self.y_min = torch.tensor(ckpt["target_minimum"], dtype=torch.float32)
            self.y_max = torch.tensor(ckpt["target_maximum"], dtype=torch.float32)

        # If your model already outputs normalized targets, set this True; otherwise False
        self.outputs_are_normalized = ckpt.get("outputs_normalized", True)

        # --- CityLearn-required attrs ---
        # Flat 2D list: [feature][ready_flag + lookback scalars]
        self._model_input  = [[0.0] + [0.0]*self.lookback for _ in range(self.n_features)]
        # Dummy LSTM hidden state (h, c)
        self._hidden_state = (torch.zeros(1,1,1), torch.zeros(1,1,1))

    def reset(self):
        self._model_input  = [[0.0] + [0.0]*self.lookback for _ in range(self.n_features)]
        self._hidden_state = (torch.zeros(1,1,1), torch.zeros(1,1,1))

    def __call__(self, x: torch.Tensor, h):
        """
        x: (lookback, n_features) float32, normalized by CityLearn
        h: (h, c) tuple (ignored, passed through)
        returns: (indoor_temp_norm: (1,1) tensor in [0,1], h_next)
        """
        # Ensure (T, F)
        if x.dim() == 3:  # (B,T,F) -> (T,F)
            x = x.squeeze(0)
        assert x.dim() == 2 and x.shape[0] == self.lookback and x.shape[1] == self.n_features, \
            f"expected {(self.lookback, self.n_features)}, got {tuple(x.shape)}"

        # Mirror x into _model_input with FLAT scalars (no nested lists)
        x_np = x.detach().cpu().float().numpy()   # (T, F)
        for i in range(self.n_features):
            row = self._model_input[i]
            row[0] = 1.0  # ready
            # fill t0..t{lookback-1}
            col = x_np[:, i].tolist()
            row[1:self.lookback+1] = [float(v) for v in col]

        # Run your model
        with torch.no_grad():
            y_all = self.model(x.unsqueeze(0)).squeeze(0)  # (num_targets,)

        y_t = y_all[self.indoor_idx]

        # Return normalized indoor temp as CityLearn expects
        if self.outputs_are_normalized:
            y_norm = torch.clamp(y_t, 0.0, 1.0)
        else:
            # model outputs °C -> normalize using checkpoint bounds
            assert self.has_y_scale, "Need target_min/max to normalize °C output."
            y_norm = (y_t - self.y_min[self.indoor_idx]) / (self.y_max[self.indoor_idx] - self.y_min[self.indoor_idx] + 1e-8)
            y_norm = torch.clamp(y_norm, 0.0, 1.0)

        return y_norm.view(1,1), h
