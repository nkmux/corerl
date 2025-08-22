"""
Data Manager for MPC Analysis System
"""

import pandas as pd
import json

class DataManager:
    """
    Handles data loading, validation, and saving operations
    """
    
    def __init__(self):
        pass
    
    def load_and_prepare_data(self):
        """Load and prepare all CSV data"""
        print("Loading building and weather data...")
        
        try:
            # Load building datasets
            building_1 = pd.read_csv('./data/building_1.csv')
            building_2 = pd.read_csv('./data/building_2.csv') 
            building_3 = pd.read_csv('./data/building_3.csv')
            
            # Load weather datasets
            weather_abha = pd.read_csv('./data/weather_abha.csv')
            weather_jeddah = pd.read_csv('./data/weather_jeddah.csv')
            weather_riyadh = pd.read_csv('./data/weather_riyadh.csv')
            
            print("Data loaded successfully!")
            print(f"Building data shapes: {building_1.shape}, {building_2.shape}, {building_3.shape}")
            print(f"Weather data shapes: {weather_abha.shape}, {weather_jeddah.shape}, {weather_riyadh.shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure CSV files are in ./data/ directory")
            return None
        
        # Data cleaning and validation
        buildings = [building_1, building_2, building_3]
        weather_data = [weather_abha, weather_jeddah, weather_riyadh]
        
        # Clean building data using forward and backward fill
        for i, building in enumerate(buildings):
            buildings[i] = building.fillna(method='ffill').fillna(method='bfill')
            
        # Clean weather data using forward and backward fill
        for i, weather in enumerate(weather_data):
            weather_data[i] = weather.fillna(method='ffill').fillna(method='bfill')
        
        # Validate data ranges for buildings
        for i, building in enumerate(buildings):
            temp_range = building['indoor_dry_bulb_temperature']
            humidity_range = building['indoor_relative_humidity']
            print(f"Building {i+1} - Temperature: {temp_range.min():.1f} to {temp_range.max():.1f}°C, "
                  f"Humidity: {humidity_range.min():.1f} to {humidity_range.max():.1f}%")
            
            # Check occupancy data
            occupancy_range = building['occupant_count']
            print(f"Building {i+1} - Occupancy: {occupancy_range.min():.0f} to {occupancy_range.max():.0f} people")
        
        # Validate data ranges for weather
        for i, weather in enumerate(weather_data):
            weather_names = ['Abha', 'Jeddah', 'Riyadh']
            temp_range = weather['outdoor_dry_bulb_temperature']
            humidity_range = weather['outdoor_relative_humidity']
            print(f"{weather_names[i]} Weather - Temperature: {temp_range.min():.1f} to {temp_range.max():.1f}°C, "
                  f"Humidity: {humidity_range.min():.0f} to {humidity_range.max():.0f}%")
            
            # Check solar irradiance data
            if 'direct_solar_irradiance' in weather.columns:
                solar_range = weather['direct_solar_irradiance']
                print(f"{weather_names[i]} Weather - Solar: {solar_range.min():.0f} to {solar_range.max():.0f} W/m²")
        
        return {
            'buildings': buildings,
            'weather': weather_data,
            'building_names': ['Building_1', 'Building_2', 'Building_3'],
            'weather_names': ['Abha', 'Jeddah', 'Riyadh']
        }
    
    def validate_data_consistency(self, data):
        """Validate that all datasets have consistent time series length"""
        if data is None:
            return False
        
        # Check that all building datasets have same length
        building_lengths = [len(df) for df in data['buildings']]
        weather_lengths = [len(df) for df in data['weather']]
        
        if len(set(building_lengths)) > 1:
            print("Warning: Building datasets have different lengths")
            print(f"Building lengths: {building_lengths}")
        
        if len(set(weather_lengths)) > 1:
            print("Warning: Weather datasets have different lengths")
            print(f"Weather lengths: {weather_lengths}")
        
        # Use minimum length to ensure compatibility
        min_length = min(min(building_lengths), min(weather_lengths))
        print(f"Using minimum data length: {min_length} hours ({min_length/24:.1f} days)")
        
        # Truncate all datasets to minimum length
        for i in range(len(data['buildings'])):
            data['buildings'][i] = data['buildings'][i].iloc[:min_length]
        for i in range(len(data['weather'])):
            data['weather'][i] = data['weather'][i].iloc[:min_length]
        
        return True
    
    def save_detailed_results(self, results_summary, output_dir):
        """Save detailed results to CSV files"""
        print("Saving detailed results...")
        
        # Prepare summary data
        summary_data = []
        seasonal_data = []
        
        for combo_key, data in results_summary.items():
            # Handle combo_key that might have multiple underscores
            parts = combo_key.split('_')
            if len(parts) >= 2:
                building = '_'.join(parts[:-1])  # Everything except last part
                weather = parts[-1]              # Last part
            else:
                building = parts[0] if parts else 'Unknown'
                weather = 'Unknown'
            annual = data['annual']
            
            # Annual summary
            summary_data.append({
                'Building': building,
                'Weather': weather,
                'Annual_Cost': annual.get('total_annual_cost', 0),
                'Avg_Violation_Rate': annual.get('avg_violation_rate', 0),
                'Avg_Energy_Consumption': annual.get('avg_energy_consumption', 0),
                'Annual_Savings': annual.get('annual_savings_vs_baseline', 0),
                'Success_Rate': annual.get('avg_success_rate', 0),
                'Seasons_Analyzed': annual.get('number_of_seasons', 0)
            })
            
            # Seasonal breakdown
            for season, metrics in data['seasons'].items():
                seasonal_data.append({
                    'Building': building,
                    'Weather': weather,
                    'Season': season,
                    'Cost': metrics.get('total_energy_cost', 0),
                    'Violation_Rate': metrics.get('violation_rate', 0),
                    'Avg_Energy': metrics.get('avg_energy_consumption', 0),
                    'Success_Rate': metrics.get('success_rate', 0)
                })
        
        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        seasonal_df = pd.DataFrame(seasonal_data)
        
        summary_df.to_csv(f'{output_dir}/data/annual_summary.csv', index=False)
        seasonal_df.to_csv(f'{output_dir}/data/seasonal_breakdown.csv', index=False)
        
        # Save detailed results as JSON for potential future analysis
        with open(f'{output_dir}/data/detailed_results.json', 'w') as f:
            # Remove detailed time series data for JSON (too large)
            simplified_results = {}
            for combo_key, data in results_summary.items():
                simplified_results[combo_key] = {
                    'building': data['building'],
                    'weather': data['weather'],
                    'annual': data['annual'],
                    'seasons': data['seasons']
                }
            json.dump(simplified_results, f, indent=2)
        
        print(f"Detailed results saved to {output_dir}/data/")
        
        # Print summary statistics
        if summary_data:
            print(f"\nSummary Statistics:")
            print(f"Total combinations analyzed: {len(summary_data)}")
            costs = [item['Annual_Cost'] for item in summary_data]
            violations = [item['Avg_Violation_Rate'] for item in summary_data]
            print(f"Cost range: ${min(costs):.0f} - ${max(costs):.0f}")
            print(f"Violation rate range: {min(violations):.1f}% - {max(violations):.1f}%")
    
    def check_required_columns(self, data):
        """Check that all required columns are present in the datasets"""
        # Required columns for building data
        required_building_cols = [
            'indoor_dry_bulb_temperature',
            'indoor_relative_humidity', 
            'occupant_count'
        ]
        
        # Required columns for weather data
        required_weather_cols = [
            'outdoor_dry_bulb_temperature',
            'outdoor_relative_humidity'
        ]
        
        # Optional weather columns
        optional_weather_cols = [
            'direct_solar_irradiance',
            'diffuse_solar_irradiance'
        ]
        
        all_valid = True
        
        # Check building datasets
        for i, building in enumerate(data['buildings']):
            missing_cols = [col for col in required_building_cols if col not in building.columns]
            if missing_cols:
                print(f"Warning: Building {i+1} missing columns: {missing_cols}")
                all_valid = False
            else:
                print(f"✓ Building {i+1} has all required columns")
        
        # Check weather datasets
        for i, weather in enumerate(data['weather']):
            weather_name = data['weather_names'][i]
            missing_cols = [col for col in required_weather_cols if col not in weather.columns]
            if missing_cols:
                print(f"Warning: {weather_name} weather missing columns: {missing_cols}")
                all_valid = False
            else:
                print(f"✓ {weather_name} weather has all required columns")
            
            # Check for optional columns
            for col in optional_weather_cols:
                if col not in weather.columns:
                    print(f"Note: {weather_name} weather missing optional column: {col}")
        
        return all_valid
    
    def preprocess_data_for_mpc(self, data):
        """Preprocess data to ensure compatibility with MPC controller"""
        if not self.check_required_columns(data):
            print("Warning: Some required columns are missing. Proceeding with available data.")
        
        if not self.validate_data_consistency(data):
            print("Error: Data validation failed")
            return None
        
        # Ensure weather data has solar radiation column (use 0 if missing)
        for i, weather in enumerate(data['weather']):
            if 'direct_solar_irradiance' not in weather.columns:
                weather['direct_solar_irradiance'] = 0
                print(f"Added default solar irradiance for {data['weather_names'][i]}")
        
        # Ensure all numeric columns are properly typed
        for i, building in enumerate(data['buildings']):
            numeric_cols = ['indoor_dry_bulb_temperature', 'indoor_relative_humidity', 'occupant_count']
            for col in numeric_cols:
                if col in building.columns:
                    data['buildings'][i][col] = pd.to_numeric(building[col], errors='coerce')
        
        for i, weather in enumerate(data['weather']):
            numeric_cols = ['outdoor_dry_bulb_temperature', 'outdoor_relative_humidity', 'direct_solar_irradiance']
            for col in numeric_cols:
                if col in weather.columns:
                    data['weather'][i][col] = pd.to_numeric(weather[col], errors='coerce')
        
        print("Data preprocessing completed successfully!")
        return data