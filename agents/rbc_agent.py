from typing import List

from citylearn.agents.rbc import BasicRBC
from citylearn.citylearn import CityLearnEnv

class RBC(BasicRBC):
    """
    Improved RBC agent with smart storage control logic.
    """
    
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)
        self.comfort_band = 1.0
        
        # Storage control parameters
        self.solar_charge_threshold = 200  # W/m² minimum solar to charge
        self.high_soc_threshold = 0.8      # Don't charge above this
        self.low_soc_threshold = 0.2       # Discharge priority below this
        self.charge_rate = 0.3             # Storage charging rate
        self.discharge_rate = -0.2         # Storage discharge rate
        
        # Peak hours definition
        self.morning_peak = (6, 9)         # 6-9 AM
        self.evening_peak = (18, 22)       # 6-10 PM

    def predict(self, observations: List[List[float]], deterministic: bool = True) -> List[List[float]]:
        actions = []

        for a, n, o in zip(self.action_names, self.observation_names, observations):
            # Extract key observations
            hour = o[n.index('hour')]
            indoor_temp = o[n.index('indoor_dry_bulb_temperature')]
            setpoint = o[n.index('indoor_dry_bulb_temperature_cooling_set_point')]
            cooling_demand = o[n.index('cooling_demand')]
            
            # Storage states
            cooling_soc = o[n.index('cooling_storage_soc')]
            dhw_soc = o[n.index('dhw_storage_soc')]
            
            # Solar and weather
            solar_irradiance = o[n.index('direct_solar_irradiance')]
            
            # Calculate temperature difference
            temp_difference = indoor_temp - setpoint
            
            actions_ = []

            for a_n in a:
                if a_n == 'cooling_device':
                    # Cooling device control 
                    if temp_difference > self.comfort_band:
                        actions_.append(0.8)
                    else:
                        actions_.append(0.0)
                
                elif a_n == 'cooling_storage':
                    action = self._control_cooling_storage(
                        hour, cooling_soc, solar_irradiance, 
                        cooling_demand, temp_difference
                    )
                    actions_.append(action)
                
                elif a_n == 'dhw_storage':
                    action = self._control_dhw_storage(
                        hour, dhw_soc, solar_irradiance
                    )
                    actions_.append(action)
                
            actions.append(actions_)
        
        return actions
    
    def _control_cooling_storage(self, hour, soc, solar, cooling_demand, temp_diff):
        """
        Smart cooling storage control logic.
        
        Priority order:
        1. Discharge during peak hours if cooling needed and storage available
        2. Charge during high solar if storage not full
        3. Discharge if indoor temp too high and storage available
        4. Otherwise, do nothing
        """
        
        # Check if it's peak hours
        is_peak = (self.morning_peak[0] <= hour <= self.morning_peak[1] or 
                   self.evening_peak[0] <= hour <= self.evening_peak[1])
        
        # High cooling demand conditions
        high_cooling_need = (cooling_demand > 0 or temp_diff > self.comfort_band)
        
        # 1. Peak hours: discharge if cooling needed and storage available
        if is_peak and high_cooling_need and soc > self.low_soc_threshold:
            return self.discharge_rate
        
        # 2. High solar: charge storage if not full
        elif (solar > self.solar_charge_threshold and 
              soc < self.high_soc_threshold):
            return self.charge_rate
        
        # 3. Emergency cooling: discharge if very hot and storage available
        elif temp_diff > (self.comfort_band + 1.0) and soc > self.low_soc_threshold:
            return self.discharge_rate * 0.5  # Gentler discharge
        
        # 4. Default: do nothing
        else:
            return 0.0
    
    def _control_dhw_storage(self, hour, soc, solar):
        """
        DHW storage control logic.
        
        Strategy:
        - Charge during high solar periods
        - Discharge during morning/evening when DHW demand typically high
        """
        
        # Morning/evening DHW demand periods
        dhw_demand_periods = (6 <= hour <= 9 or 18 <= hour <= 22)
        
        # Charge during high solar if storage not full
        if (solar > self.solar_charge_threshold and 
            soc < self.high_soc_threshold):
            return self.charge_rate
        
        # Discharge during typical DHW demand periods if storage available
        elif dhw_demand_periods and soc > self.low_soc_threshold:
            return self.discharge_rate * 0.7  # Gentler discharge for DHW
        
        # Default: do nothing
        else:
            return 0.0
