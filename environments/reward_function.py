from typing import Any, List, Mapping, Union
import numpy as np

from citylearn.reward_function import ComfortReward


class ComfortandConsumptionReductionReward(ComfortReward):
    def __init__(self, env_metadata: Mapping[str, Any]):
        self.multiplier = 3.0          # extra penalty when too cold
        self.deadband  = 0.5           # °C tolerance around setpoint
        super().__init__(env_metadata)
        self._prev_p: List[Union[float, None]] = []  # per-building P_{t-1}

    def _ensure_state(self, n_buildings: int):
        if len(self._prev_p) != n_buildings:
            self._prev_p = [None] * n_buildings

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        n = len(observations)
        self._ensure_state(n)
        rewards: List[float] = []

        for i, o in enumerate(observations):
            Tin   = float(o['indoor_dry_bulb_temperature'])
            Tset  = float(o['indoor_dry_bulb_temperature_cooling_set_point'])
            P     = float(o['net_electricity_consumption'])
            outage = int(o.get('power_outage', 0))
            nsl    = float(o.get('non_shiftable_load', 0.0))

            # --- Comfort with deadband + cold multiplier
            delta = Tin - Tset
            err   = max(0.0, abs(delta) - self.deadband)
            if delta < 0.0:  # too cold
                err *= self.multiplier

            # --- Ramping: time-difference per building
            prevP = self._prev_p[i]
            ramp  = abs(P - prevP) if prevP is not None else 0.0
            self._prev_p[i] = P

            # --- Outage unserved energy (nonnegative and zero if no outage)
            s = max(0.0, nsl - P) if outage else 0.0

            # NOTE: weights are in different units; tune/normalize as needed
            reward = -0.3 * err - 0.15 * ramp - 0.1 * s
            rewards.append(reward)

        return [float(np.mean(rewards))] if self.central_agent else rewards
