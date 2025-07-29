import pandas as pd

# Load the uploaded EPW file
epw_path = '../data/raw/SAU_RI_Riyadh-Khalid.Intl.AP.404370_TMYx.2009-2023.epw'
weather = pd.read_csv(epw_path, skiprows=8, header=None)

# The CORRECT weather columns, based on the EPW format
weather.columns = [
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

# Select required columns
selected = weather[['DryBulbTemp_C', 'RelativeHumidity_%',
                    'DiffuseHorizontalRad_W/m2', 'DirectNormalRad_W/m2']]

# Compute rolling averages
features = pd.DataFrame()
features['Outdoor Drybulb Temperature (C)'] = selected['DryBulbTemp_C']
features['Outdoor Relative Humidity (%)'] = selected['RelativeHumidity_%']
features['Diffuse Solar Radiation (W/m2)'] = selected['DiffuseHorizontalRad_W/m2']
features['Direct Solar Radiation (W/m2)'] = selected['DirectNormalRad_W/m2']

# Add rolling means
for hrs in [6, 12, 24]:
    features[f'{hrs}h Outdoor Drybulb Temperature (C)'] = selected['DryBulbTemp_C'].rolling(window=hrs, min_periods=1).mean()
    features[f'{hrs}h Outdoor Relative Humidity (%)'] = selected['RelativeHumidity_%'].rolling(window=hrs, min_periods=1).mean()
    features[f'{hrs}h Diffuse Solar Radiation (W/m2)'] = selected['DiffuseHorizontalRad_W/m2'].rolling(window=hrs, min_periods=1).mean()
    features[f'{hrs}h Direct Solar Radiation (W/m2)'] = selected['DirectNormalRad_W/m2'].rolling(window=hrs, min_periods=1).mean()

output_path = '../data/processed/weather.csv'
features.to_csv(output_path, index=False)
