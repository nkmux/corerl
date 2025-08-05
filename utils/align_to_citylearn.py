"""
This scripts aligns the output of DesignBuilder building simulation to the expected input into CityLearn.
"""
import pandas as pd
import numpy as np

def structured_dhw(hour, day_type):
    """Generate realistic DHW demand in kWh based on hour and day type."""
    
    # Determine baseline probability of usage and volume distribution parameters
    if 5 <= hour <= 9:  # Morning routine
        usage_prob = 0.85 if day_type == 0 else 0.7  
        volume_mean = 0.7
        volume_sigma = 0.4
    elif 12 <= hour <= 14:  # Lunch
        usage_prob = 0.4 if day_type == 0 else 0.5
        volume_mean = 0.3
        volume_sigma = 0.25
    elif 18 <= hour <= 22:  # Evening usage
        usage_prob = 0.9 if day_type == 0 else 0.95
        volume_mean = 0.9
        volume_sigma = 0.5
    else:  # Idle hours
        usage_prob = 0.05 if day_type == 0 else 0.08
        volume_mean = 0.02
        volume_sigma = 0.01

    # Whether demand occurs this hour
    if np.random.rand() < usage_prob:
        # Generate demand volume using lognormal distribution
        dhw_kwh = np.random.lognormal(mean=volume_mean, sigma=volume_sigma)
        return dhw_kwh
    
    return np.random.normal(0.01, 0.005)  # idle baseline


def load_and_transform_building(path: str, b_number: int):
    df = pd.read_csv(path, header=0)
    df = df.drop(index=0).reset_index(drop=True) # Drop units row

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['Date/Time'], format="mixed")
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_type'] = (df['datetime'].dt.weekday >= 5).astype(int)

    # Calculate non-shiftable load from internal loads
    df['non_shiftable_load'] = (
        df['Room Electricity'].astype(float).fillna(0) +
        df['Lighting'].astype(float).fillna(0) +
        df['Computer + Equip'].astype(float).fillna(0)
    )

    # Daily and hourly shaping
    daily_scaling = np.random.normal(1.0, 0.05, size=(len(df) // 24))
    hourly_shape = np.array([
        0.5 if 0 <= h <= 5 else
        0.8 if 6 <= h <= 8 else
        0.6 if 9 <= h <= 16 else
        1.0 if 17 <= h <= 22 else
        0.7 for h in range(24)
    ])
    df['non_shiftable_load'] *= np.repeat(daily_scaling, 24)[:len(df)]
    df['non_shiftable_load'] *= hourly_shape[df['hour'].values]

    # Overwrite or smooth dhw_demand
    df['dhw_demand'] = [
        structured_dhw(h, dt) for h, dt in zip(df['hour'], df['day_type'])
    ]
    # Final CityLearn-aligned DataFrame
    citylearn_df = pd.DataFrame({
        'month': df['month'],
        'day': df['day'],
        'hour': df['hour'],
        'day_type': df['day_type'],
        'cooling_demand': df['Cooling (Electricity)'].astype(float),
        'dhw_demand': df['dhw_demand'].astype(float),
        'non_shiftable_load': df['non_shiftable_load'],
        'occupant_count': df['Occupancy'].astype(float),
        'indoor_dry_bulb_temperature': df['Operative Temperature'].astype(float),
        'indoor_relative_humidity': df['Relative Humidity'].astype(float)
    })

    citylearn_df.to_csv(f"../data/processed/building_{b_number}.csv", index=False)
    print(f"Processed building_{b_number} and saved it.")

if __name__ == "__main__":
    load_and_transform_building("../data/raw/RUH_RES_BUILDING.csv", 1)
    load_and_transform_building("../data/raw/JED_RES_BUILDING.csv", 2)
    load_and_transform_building("../data/raw/ABH_RES_BUILDING.csv", 3)