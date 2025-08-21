from typing import Any, List, Mapping, Union
from rewards.comfort_reward import ComfortRewardFunction

class ImprovedReward(ComfortRewardFunction):
    """
    Improved reward function that includes carbon emissions optimization.
    Addresses the main weakness of the current model.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any]):
        self.multiplier = 3.0
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        total_building_consumption_minus1 = 0 
        r = 0 
        s = 0 
        g = 0  # Carbon emissions

        for o in observations:
            # Thermal comfort (same as before)
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            delta = indoor_dry_bulb_temperature - set_point

            if delta < 0.0:
                delta *= self.multiplier
            
            # Ramping (same as before)
            total_building_consumption = o['net_electricity_consumption']
            if total_building_consumption_minus1 != 0: 
                r = total_building_consumption - total_building_consumption_minus1 
            total_building_consumption_minus1 = total_building_consumption
            
            # NEW: Carbon emissions optimization
            carbon_intensity = o['carbon_intensity']
            g = max(0, total_building_consumption * carbon_intensity)
            
            # Unserved energy (same as before)
            outage = o['power_outage'] 
            non_shiftable_load = o['non_shiftable_load']

            if outage == 1: 
                s_expected = non_shiftable_load 
                s_served = total_building_consumption
                s = s_expected - s_served
            else: 
                s = 0

            # Improved reward function with carbon emissions
            # Weights adjusted based on competition importance
            comfort_penalty = -0.25 * abs(delta)  # Reduced from -0.3
            carbon_penalty = -0.2 * g  # NEW: Carbon emissions penalty
            unserved_penalty = -0.1 * s  # Same as before
            ramping_penalty = -0.15 * abs(r)  # Same as before
            
            reward = comfort_penalty + carbon_penalty + unserved_penalty + ramping_penalty
            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)/len(reward_list)]
        else:
            reward = reward_list

        return reward 