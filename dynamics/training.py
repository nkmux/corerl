from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DynamicsDataset
from .model import TransformerDynamicsModel
from .utils import add_time_cyc, chronological_split, compute_soc
from .definitions import TARGET_NAMES, INPUT_NAMES

@dataclass
class Config:
    building_csv: str = "data/building_1.csv"
    weather_csv: str = "data/weather_riyadh.csv"
    out_ckpt: str = "data/building_1.pth"
    lookback: int = 24
    batch_size: int = 256
    lr: float = 2e-3
    epochs: int = 30
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    dropout: float = 0.1
    use_fp16: bool = True
    train_frac: float = 0.75
    val_frac: float = 0.10

CONFIG = Config()

def main():
    bld = pd.read_csv(CONFIG.building_csv)
    wth = pd.read_csv(CONFIG.weather_csv)
    df = pd.concat([bld, wth], axis=1)

    # Make cyclic features
    df = add_time_cyc(df)
    df = compute_soc(df, have_cols=("cooling_storage_soc" in df and "dhw_storage_soc" in df))

    # Filter to rows without NaNs in used columns
    # df = df.dropna(subset=set(INPUT_NAMES+TARGET_NAMES)).reset_index(drop=True)

    # Chronological split
    train_df, val_df, test_df = chronological_split(df, CONFIG.lookback, CONFIG.train_frac, CONFIG.val_frac)

    # Datasets (fit scaling on train; reuse on val/test)
    train_ds = DynamicsDataset(train_df, INPUT_NAMES, TARGET_NAMES, CONFIG.lookback, fit_scale=True)

    val_ds   = DynamicsDataset(val_df,   INPUT_NAMES, TARGET_NAMES, CONFIG.lookback,
                               x_min=train_ds.x_min, x_max=train_ds.x_max,
                               y_min=train_ds.y_min, y_max=train_ds.y_max)
    test_ds  = DynamicsDataset(test_df,  INPUT_NAMES, TARGET_NAMES, CONFIG.lookback,
                               x_min=train_ds.x_min, x_max=train_ds.x_max,
                               y_min=train_ds.y_min, y_max=train_ds.y_max)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG.batch_size, shuffle=False)

    # Model
    F = len(INPUT_NAMES); TGT = len(TARGET_NAMES)
    model = TransformerDynamicsModel(F, TGT, d_model=CONFIG.d_model, nhead=CONFIG.nhead, num_layers=CONFIG.layers, dropout=CONFIG.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG.use_fp16 and device.type=="cuda"))

    best_val = float("inf"); best_state = None; patience=6; bad=0
    for epoch in range(1, CONFIG.epochs+1):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(CONFIG.use_fp16 and device.type=="cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        # val
        model.eval(); val_loss=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        print(f"epoch {epoch:02d} | train {tr_loss:.5f} | val {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Test MAE per target
    model.load_state_dict(best_state)
    model.eval()
    def mae(loader):
        se = torch.zeros(TGT); n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).cpu()
                se += (pred - yb.cpu()).abs().sum(dim=0)
                n += yb.size(0)
        return (se / n).numpy()

    mae_val  = mae(val_loader)
    mae_test = mae(test_loader)
    print("Val MAE (scaled): ", mae_val)
    print("Test MAE (scaled):", mae_test)

    # Save self-describing checkpoint (NOTE: we store x/y min/max used during scaling)
    ckpt = {
        "model_state_dict": best_state,
        "arch": {
            "input_dim": F, "target_dim": TGT,
            "d_model": CONFIG.d_model, "nhead": CONFIG.nhead, "num_layers": CONFIG.layers, "dropout": CONFIG.dropout,
        },
        "lookback": CONFIG.lookback,
        "input_observation_names": INPUT_NAMES,
        "input_normalization_minimum": train_ds.x_min.tolist(),
        "input_normalization_maximum": train_ds.x_max.tolist(),
        "target_minimum": train_ds.y_min.tolist(),
        "target_maximum": train_ds.y_max.tolist(),
    }
    torch.save(ckpt, CONFIG.out_ckpt)
    print(f"Saved checkpoint → {CONFIG.out_ckpt}")

if __name__ == "__main__":
    main()
