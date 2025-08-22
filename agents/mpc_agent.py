from typing import List, Dict, Optional
import numpy as np
import sys
import os
from pathlib import Path

# Add the mpc_benchmark-main directory to path to import the sophisticated MPC
project_root = Path(__file__).parent.parent
mpc_path = project_root / "mpc_benchmark-main"
sys.path.insert(0, str(mpc_path))

from building_mpc import BuildingEnergyMPC, SaudiElectricityPricing
from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv

class MPC(Agent):
    """
    Real Model Predictive Control (MPC) agent using the sophisticated MPC backend.
    
    This agent integrates the advanced BuildingEnergyMPC controller from mpc_benchmark-main
    with the CityLearn environment interface.
    """
    
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        
        # Initialize the sophisticated MPC controller with ORIGINAL parameters
        self.mpc_controller = BuildingEnergyMPC(
            prediction_horizon=24,      # Original: 24 steps ahead
            control_horizon=12,         # Original: 12 control moves
            sampling_time=1.0,
            comfort_temp_range=(20.0, 26.0),
            comfort_humidity_range=(35.0, 70.0),
            use_saudi_pricing=True,
            penalty_comfort=2000.0,     # Original: 2000.0
            penalty_control_effort=0.1, # Original: 0.1
            hvac_efficiency=3.2,        # From CityLearn schema
            include_storage=True,       # Enable storage optimization
            safety_factor=1.1
        )
        
        # State tracking for the MPC
        self.previous_state = {}
        self.previous_control = {}
        
        # Performance tracking
        self.mpc_success_count = 0
        self.total_calls = 0
        
    def predict(self, observations: List[List[float]], deterministic: bool = True) -> List[List[float]]:
        """
        Main prediction method that uses the sophisticated MPC backend
        """
        actions = []
        
        for building_idx, (a, n, o) in enumerate(zip(self.action_names, self.observation_names, observations)):
            self.total_calls += 1
            
            # Extract current state from CityLearn observations
            current_state = self._extract_state_from_observations(o, n, building_idx)
            
            # Generate forecasts for MPC
            weather_forecast = self._generate_weather_forecast(o, n)
            occupancy_forecast = self._generate_occupancy_forecast(o, n)
            building_demands_forecast = self._generate_demands_forecast(o, n)
            
            # Solve MPC optimization using the sophisticated backend
            try:
                optimal_control, mpc_info = self.mpc_controller.solve_mpc(
                    current_state=current_state,
                    weather_forecast=weather_forecast,
                    occupancy_forecast=occupancy_forecast,
                    building_demands_forecast=building_demands_forecast
                )
                
                if mpc_info['success']:
                    self.mpc_success_count += 1
                else:
                    # Debug: print why optimization failed (only first few times)
                    if self.total_calls <= 3:
                        print(f"[DEBUG] MPC optimization failed at call {self.total_calls}: {mpc_info.get('message', 'Unknown reason')}")
                    
            except Exception as e:
                # Debug: print exception (only first few times)  
                if self.total_calls <= 3:
                    print(f"[DEBUG] MPC exception at call {self.total_calls}: {str(e)}")
                # Fallback to simple control
                optimal_control = self._fallback_control(current_state)
                mpc_info = {'success': False, 'error': str(e)}
            
            # Convert MPC solution to CityLearn actions
            building_actions = self._convert_to_citylearn_actions(optimal_control, a)
            actions.append(building_actions)
            
            # Store state for next iteration
            self.previous_state[building_idx] = current_state
            self.previous_control[building_idx] = optimal_control
            
        return actions
    
    def _extract_state_from_observations(self, obs: List[float], obs_names: List[str], building_idx: int) -> Dict:
        """Extract building state from CityLearn observations for the MPC backend"""
        state = {}
        
        # Map CityLearn observations to MPC state format
        for i, name in enumerate(obs_names):
            if name == 'indoor_dry_bulb_temperature':
                state['temperature'] = obs[i]
            elif name == 'indoor_relative_humidity':
                state['humidity'] = obs[i]
            elif name == 'cooling_storage_soc':
                state['cooling_storage_soc'] = obs[i]
            elif name == 'dhw_storage_soc':
                state['dhw_storage_soc'] = obs[i]
            elif name == 'outdoor_dry_bulb_temperature':
                state['outdoor_temperature'] = obs[i]
            elif name == 'outdoor_relative_humidity':
                state['outdoor_humidity'] = obs[i]
            elif name == 'direct_solar_irradiance':
                state['solar_irradiance'] = obs[i]
            elif name == 'diffuse_solar_irradiance':
                state['diffuse_solar_irradiance'] = obs[i]
            elif name == 'hour':
                state['hour'] = obs[i]
            elif name == 'day_type':
                state['day_type'] = obs[i]
            elif name == 'month':
                state['month'] = obs[i]
            elif name == 'cooling_demand':
                state['cooling_demand'] = obs[i]
            elif name == 'dhw_demand':
                state['dhw_demand'] = obs[i]
            elif name == 'non_shiftable_load':
                state['non_shiftable_load'] = obs[i]
            elif name == 'occupant_count':
                state['occupant_count'] = obs[i]
            elif name == 'electricity_pricing':
                state['electricity_pricing'] = obs[i]
        
        # Set defaults for missing values
        state.setdefault('temperature', 22.0)
        state.setdefault('humidity', 50.0)
        state.setdefault('mass_temperature', state['temperature'])  # MPC needs this
        state.setdefault('cooling_storage_soc', 0.5)
        state.setdefault('dhw_storage_soc', 0.5)
        state.setdefault('outdoor_temperature', 30.0)
        state.setdefault('outdoor_humidity', 60.0)
        state.setdefault('solar_irradiance', 0.0)
        state.setdefault('diffuse_solar_irradiance', 0.0)
        state.setdefault('hour', 12.0)
        state.setdefault('day_type', 1.0)
        state.setdefault('month', 6.0)
        state.setdefault('cooling_demand', 0.0)
        state.setdefault('dhw_demand', 0.0)
        state.setdefault('non_shiftable_load', 0.5)
        state.setdefault('occupant_count', 2.0)
        state.setdefault('electricity_pricing', 0.18)
        
        # Add previous control values if available
        if building_idx in self.previous_control:
            prev_ctrl = self.previous_control[building_idx]
            state['prev_T_set'] = prev_ctrl.get('temperature_setpoint', 22.0)
            state['prev_RH_set'] = prev_ctrl.get('humidity_setpoint', 50.0)
            state['prev_cooling_storage'] = prev_ctrl.get('cooling_storage_charge', 0.0)
            state['prev_dhw_storage'] = prev_ctrl.get('dhw_storage_charge', 0.0)
            state['prev_cooling_device'] = prev_ctrl.get('cooling_device_intensity', 1.0)
        
        return state
    
    def _generate_weather_forecast(self, obs: List[float], obs_names: List[str]) -> List[Dict]:
        """Generate weather forecast for MPC horizon"""
        # Extract current weather
        current_outdoor_temp = 30.0
        current_outdoor_humidity = 60.0
        current_solar = 0.0
        current_diffuse_solar = 0.0
        current_hour = 12.0
        
        for i, name in enumerate(obs_names):
            if name == 'outdoor_dry_bulb_temperature':
                current_outdoor_temp = obs[i]
            elif name == 'outdoor_relative_humidity':
                current_outdoor_humidity = obs[i]
            elif name == 'direct_solar_irradiance':
                current_solar = obs[i]
            elif name == 'diffuse_solar_irradiance':
                current_diffuse_solar = obs[i]
            elif name == 'hour':
                current_hour = obs[i]
        
        # Generate realistic forecast with daily patterns
        forecast = []
        for k in range(self.mpc_controller.N_pred):
            hour = (current_hour + k) % 24
            
            # Simple sinusoidal temperature pattern
            temp_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 2 PM
            outdoor_temp = current_outdoor_temp + temp_variation
            
            # Simple solar pattern (zero at night)
            if 6 <= hour <= 18:
                solar_factor = np.sin(np.pi * (hour - 6) / 12) ** 2
                solar = current_solar * solar_factor
                diffuse_solar = current_diffuse_solar * solar_factor
            else:
                solar = 0.0
                diffuse_solar = 0.0
            
            forecast.append({
                'temperature': outdoor_temp,           # MPC backend expects 'temperature'
                'humidity': current_outdoor_humidity,  # MPC backend expects 'humidity'
                'solar_radiation': solar,              # MPC backend expects 'solar_radiation' 
                'diffuse_solar_irradiance': diffuse_solar,
                'hour': hour
            })
        
        return forecast
    
    def _generate_occupancy_forecast(self, obs: List[float], obs_names: List[str]) -> List[float]:
        """Generate occupancy forecast"""
        current_hour = 12.0
        current_occupants = 2.0
        
        for i, name in enumerate(obs_names):
            if name == 'hour':
                current_hour = obs[i]
            elif name == 'occupant_count':
                current_occupants = obs[i]
        
        # Generate occupancy pattern
        forecast = []
        for k in range(self.mpc_controller.N_pred):
            hour = (current_hour + k) % 24
            if 8 <= hour <= 18:  # Daytime
                occupancy = current_occupants
            elif 18 <= hour <= 22:  # Evening
                occupancy = current_occupants * 1.2  # Higher evening activity
            else:  # Night
                occupancy = current_occupants * 0.5  # Lower night activity
            
            forecast.append(occupancy)
        
        return forecast
    
    def _generate_demands_forecast(self, obs: List[float], obs_names: List[str]) -> List[Dict]:
        """Generate building demands forecast"""
        current_cooling = 0.0
        current_dhw = 0.0
        current_non_shiftable = 0.5
        current_hour = 12.0
        
        for i, name in enumerate(obs_names):
            if name == 'cooling_demand':
                current_cooling = obs[i]
            elif name == 'dhw_demand':
                current_dhw = obs[i]
            elif name == 'non_shiftable_load':
                current_non_shiftable = obs[i]
            elif name == 'hour':
                current_hour = obs[i]
        
        # Generate demand patterns
        forecast = []
        for k in range(self.mpc_controller.N_pred):
            hour = (current_hour + k) % 24
            
            # DHW demand pattern (peaks in morning and evening)
            if 6 <= hour <= 9 or 18 <= hour <= 22:
                dhw_factor = 1.5
            else:
                dhw_factor = 0.5
            
            # Cooling demand varies with temperature and time
            if 10 <= hour <= 18:  # Daytime cooling
                cooling_factor = 1.2
            else:
                cooling_factor = 0.8
            
            forecast.append({
                'cooling_demand': current_cooling * cooling_factor,
                'dhw_demand': current_dhw * dhw_factor,
                'non_shiftable_load': current_non_shiftable
            })
        
        return forecast
    
    def _convert_to_citylearn_actions(self, optimal_control: Dict, action_names: List[str]) -> List[float]:
        """Convert MPC optimal control to CityLearn action format"""
        actions = []
        
        for action_name in action_names:
            if action_name == 'cooling_device':
                # Use cooling device intensity from MPC
                actions.append(optimal_control.get('cooling_device_intensity', 0.5))
            elif action_name == 'cooling_storage':
                # Use cooling storage charge command from MPC
                actions.append(optimal_control.get('cooling_storage_charge', 0.0))
            elif action_name == 'dhw_storage':
                # Use DHW storage charge command from MPC
                actions.append(optimal_control.get('dhw_storage_charge', 0.0))
            else:
                # Default for unknown actions
                actions.append(0.0)
        
        return actions
    
    def _fallback_control(self, current_state: Dict) -> Dict:
        """Fallback control when MPC optimization fails"""
        T_indoor = current_state['temperature']
        cooling_soc = current_state['cooling_storage_soc']
        dhw_soc = current_state['dhw_storage_soc']
        hour = current_state['hour']
        solar = current_state['solar_irradiance']
        
        # Simple rule-based fallback
        if T_indoor > 24.0:
            cooling_device = 1.0
        elif T_indoor > 22.5:
            cooling_device = 0.6
        else:
            cooling_device = 0.2
        
        # Storage control
        is_peak = (6 <= hour <= 9) or (18 <= hour <= 22)
        
        if solar > 200 and cooling_soc < 0.8:
            cooling_storage = 0.4
        elif is_peak and cooling_soc > 0.3:
            cooling_storage = -0.3
        else:
            cooling_storage = 0.0
        
        if solar > 300 and dhw_soc < 0.8:
            dhw_storage = 0.3
        elif is_peak and dhw_soc > 0.3:
            dhw_storage = -0.2
        else:
            dhw_storage = 0.0
        
        return {
            'temperature_setpoint': 22.0,
            'humidity_setpoint': 50.0,
            'cooling_device_intensity': cooling_device,
            'cooling_storage_charge': cooling_storage,
            'dhw_storage_charge': dhw_storage
        }
    
    def get_performance_stats(self) -> Dict:
        """Get MPC performance statistics"""
        success_rate = self.mpc_success_count / max(self.total_calls, 1) * 100
        return {
            'mpc_success_rate': success_rate,
            'total_calls': self.total_calls,
            'successful_optimizations': self.mpc_success_count
        } 