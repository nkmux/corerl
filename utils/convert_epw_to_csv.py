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

def process_weather(df, city_name):
    selected = df[['DryBulbTemp_C', 'RelativeHumidity_%',
                   'DiffuseHorizontalRad_W/m2', 'DirectNormalRad_W/m2']].copy()
    features = pd.DataFrame()
    
    features['outdoor_dry_bulb_temperature'] = selected['DryBulbTemp_C']
    features['outdoor_relative_humidity'] = selected['RelativeHumidity_%']
    features['diffuse_solar_irradiance'] = selected['DiffuseHorizontalRad_W/m2']
    features['direct_solar_irradiance'] = selected['DirectNormalRad_W/m2']

    for idx, hrs in enumerate([6, 12, 24], start=1):
        features[f'outdoor_dry_bulb_temperature_predicted_{idx}'] = (
            selected['DryBulbTemp_C'].rolling(window=hrs, min_periods=1).mean()
        )
        features[f'outdoor_relative_humidity_predicted_{idx}'] = (
            selected['RelativeHumidity_%'].rolling(window=hrs, min_periods=1).mean()
        )
        features[f'diffuse_solar_irradiance_predicted_{idx}'] = (
            selected['DiffuseHorizontalRad_W/m2'].rolling(window=hrs, min_periods=1).mean()
        )
        features[f'direct_solar_irradiance_predicted_{idx}'] = (
            selected['DirectNormalRad_W/m2'].rolling(window=hrs, min_periods=1).mean()
        )

    # Save
    features['hour'] = df['Hour']
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
