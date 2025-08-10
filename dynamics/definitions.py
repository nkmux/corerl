TARGET_NAMES = [
    "cooling_device",
    "dhw_device",
    "cooling_storage_soc",
    "dhw_storage_soc",
    "indoor_dry_bulb_temperature",
]

INPUT_NAMES = [
    # weather & loads (rename to match your columns)
    "direct_solar_irradiance",
    "diffuse_solar_irradiance",
    "outdoor_dry_bulb_temperature",
    "occupant_count",
    "cooling_demand",
    # time encodings
    "month_sin","month_cos","hour_sin","hour_cos","day_type_sin","day_type_cos",
    # current indoor temperature (autoregressive feature)
    "indoor_dry_bulb_temperature",
    # (optional) previous actions could be included as inputs; we'll test with and without
    # "cooling_device", "dhw_device",
    # "cooling_storage_soc", "dhw_storage_soc",
]