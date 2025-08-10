import numpy as np
import pandas as pd
    
def add_time_cyc(df: pd.DataFrame):
    """
    Adds time
    """
    df = df.copy()
    df["month_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    # weekend/weekday to sin/cos (or keep raw 0/1)
    df["day_type_sin"] = np.sin(np.pi*df["day_type"])
    df["day_type_cos"] = np.cos(np.pi*df["day_type"])
    return df

def chronological_split(df: pd.DataFrame, lookback: int, train_frac=0.75, val_frac=0.10):
    DF_LEN = len(df)
    train_end = int(DF_LEN * train_frac)
    val_end   = int(DF_LEN * (train_frac + val_frac))
    # ensure we have enough room for lookback at boundaries
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end-lookback:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end-lookback:].reset_index(drop=True)
    return train_df, val_df, test_df

def compute_soc(df, have_cols=True, cool_cap_kwh=70.0, dhw_cap_kwh=12.0):
    """Create cooling_storage_soc, dhw_storage_soc if missing; very simple rule-based."""
    df = df.copy()
    if have_cols and all(c in df.columns for c in ["cooling_storage_soc","dhw_storage_soc"]):
        return df
    # If actions missing, assume small actions so SOC just idles
    df["cooling_device"] = df.get("cooling_device", pd.Series(0.0, index=df.index))
    df["dhw_device"]     = df.get("dhw_device", pd.Series(0.0, index=df.index))

    soc_cool = 0.5*cool_cap_kwh
    soc_dhw  = 0.6*dhw_cap_kwh
    cool_soc, dhw_soc = [], []
    for _, r in df.iterrows():
        # toy policy: charge at night, discharge a bit during evening peaks
        if 0 <= int(r["hour"]) <= 5:
            soc_cool = min(cool_cap_kwh, soc_cool + 0.4)  # 0.4 kWh/hr
            soc_dhw  = min(dhw_cap_kwh,  soc_dhw  + 0.2)
        if 18 <= int(r["hour"]) <= 22:
            soc_cool = max(0.0, soc_cool - 0.5)
            soc_dhw  = max(0.0, soc_dhw  - 0.3)
        cool_soc.append(soc_cool/cool_cap_kwh if cool_cap_kwh>0 else 0.0)
        dhw_soc.append(soc_dhw/dhw_cap_kwh if dhw_cap_kwh>0 else 0.0)
    df["cooling_storage_soc"] = cool_soc
    df["dhw_storage_soc"]     = dhw_soc
    return df
