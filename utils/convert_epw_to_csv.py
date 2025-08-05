"""
This script converts epw files to the csv format accepted by the CityLearn library.
It uses the files in the data/raw directory, processes them, and put the processed versions in data/processed. 
"""
import pandas as pd
import numpy as np
import os

# Load EPW files
def load_weather(path):
    df = pd.read_csv(path, skiprows=8, header=None)
    df.columns = [
        'Year', 'Month', 'Day', 'Hour', 'Minute', 'DataSource',
        'DryBulbTemp_C', 'DewPointTemp_C', 'RelativeHumidity_%', 'AtmosPressure_Pa',
        'ExtraterrestrialHorzRad_W/m2', 'ExtraterrestrialDirectNormRad_W/m2',
        'GlobalHorizontalRad_W/m2', 'DirectNormalRad_W/m2', 'DiffuseHorizontalRad_W/m2',
        'GlobalHorizontalIlluminance_lux', 'DirectNormalIlluminance_lux',
        'DiffuseHorizontalIlluminance_lux', 'ZenithLuminance_lux',
        'WindDirection_deg', 'WindSpeed_m/s', 'TotalSkyCover', 'OpaqueSkyCover',
        'Visibility_km', 'CeilingHeight_m', 'PresentWeatherObs', 'PresentWeatherCodes',
        'PrecipitableWater_mm', 'AerosolOpticalDepth', 'SnowDepth_cm', 'DaysSinceLastSnow',
        'Albedo', 'LiquidPrecipDepth_mm', 'LiquidPrecipRate_mm/hr', 'Extension'
    ]
    return df

# Generate features and simulate outages
def process_weather(df, city_name, outage_hours_per_year=200, outage_duration_mean=2):
    selected = df[['DryBulbTemp_C', 'RelativeHumidity_%',
                   'DiffuseHorizontalRad_W/m2', 'DirectNormalRad_W/m2']].copy()

    features = pd.DataFrame()
    features['Outdoor Drybulb Temperature (C)'] = selected['DryBulbTemp_C']
    features['Outdoor Relative Humidity (%)'] = selected['RelativeHumidity_%']
    features['Diffuse Solar Radiation (W/m2)'] = selected['DiffuseHorizontalRad_W/m2']
    features['Direct Solar Radiation (W/m2)'] = selected['DirectNormalRad_W/m2']

    for hrs in [6, 12, 24]:
        features[f'{hrs}h Outdoor Drybulb Temperature (C)'] = selected['DryBulbTemp_C'].rolling(window=hrs, min_periods=1).mean()
        features[f'{hrs}h Outdoor Relative Humidity (%)'] = selected['RelativeHumidity_%'].rolling(window=hrs, min_periods=1).mean()
        features[f'{hrs}h Diffuse Solar Radiation (W/m2)'] = selected['DiffuseHorizontalRad_W/m2'].rolling(window=hrs, min_periods=1).mean()
        features[f'{hrs}h Direct Solar Radiation (W/m2)'] = selected['DirectNormalRad_W/m2'].rolling(window=hrs, min_periods=1).mean()

    # Simulate outages
    n = len(features)
    outage_flag = np.zeros(n, dtype=int)

    total_outage_hours = outage_hours_per_year * ((n // 8760) or 1)
    outage_starts = np.random.choice(np.arange(n), size=total_outage_hours // outage_duration_mean, replace=False)

    for start in outage_starts:
        duration = int(np.random.exponential(outage_duration_mean))
        outage_flag[start:min(start + duration, n)] = 1

    features['Power Outage Flag'] = outage_flag

    # Save
    output_path = f'../data/weather_{city_name}.csv'
    features.to_csv(output_path, index=False)
    print(f"Processed: {city_name}, to: {output_path}")

# Paths
weather_paths = {
    'abha': '../data/raw/SAU_AS_Khalid.AB.411140_TMYx.2009-2023.epw',
    'riyadh': '../data/raw/SAU_RI_Riyadh-Khalid.Intl.AP.404370_TMYx.2009-2023.epw',
    'jeddah': '../data/raw/SAU_MK_Jeddah-Abdulaziz.Intl.AP.410240_TMYx.2009-2023.epw',
}

# Process all cities
for city, path in weather_paths.items():
    df = load_weather(path)
    process_weather(df, city)
