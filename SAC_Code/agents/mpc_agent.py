from agents.rbc_agent import BasicRBCAgent
import numpy as np

class SimpleMPC(BasicRBCAgent):
    """
    Simple MPC-like baseline:
    - Proportional cooling control to keep comfort while avoiding overreaction
    - Storage scheduling: charge on solar, discharge during peak hours, maintain mid-SOC otherwise
    This is deterministic and fast, intended as a stronger baseline than RBC.
    """
    def __init__(self, env: 'local_evaluation.WrapperEnv'):
        super().__init__(env)
        self.comfort_band = 0.5   # tighter than RBC
        self.cooling_gain = 4.0   # proportional gain for cooling power
        self.max_storage_rate = 0.3
        self.soc_target_peak_discharge = 0.3
        self.soc_target_solar_charge = 0.9
        self.soc_target_idle = 0.6

    def _is_peak_hour(self, hour: float) -> bool:
        return (6 <= hour <= 9) or (18 <= hour <= 22)

    def predict(self, observations: 'list[list[float]]') -> 'list[list[float]]':
        actions = []

        for action_names, obs_names, obs in zip(self.action_names, self.observation_names, observations):
            hour = obs[obs_names.index('hour')] if 'hour' in obs_names else 12.0
            Tin = obs[obs_names.index('indoor_dry_bulb_temperature')]
            if 'indoor_dry_bulb_temperature_set_point' in obs_names:
                Tsp = obs[obs_names.index('indoor_dry_bulb_temperature_set_point')]
            elif 'indoor_dry_bulb_temperature_cooling_set_point' in obs_names:
                Tsp = obs[obs_names.index('indoor_dry_bulb_temperature_cooling_set_point')]
            else:
                Tsp = 22.0
            solar = obs[obs_names.index('solar_generation')] if 'solar_generation' in obs_names else 0.0
            soc_cool = obs[obs_names.index('cooling_storage_soc')] if 'cooling_storage_soc' in obs_names else 0.5
            soc_dhw = obs[obs_names.index('dhw_storage_soc')] if 'dhw_storage_soc' in obs_names else 0.5

            a = []
            for a_name in action_names:
                if a_name == 'cooling_device':
                    # Proportional cooling: ramp power with violation above comfort band
                    violation = Tin - Tsp - self.comfort_band
                    power = 0.0 if violation <= 0 else min(1.0, violation / self.cooling_gain)
                    a.append(power)
                elif a_name == 'cooling_storage':
                    if solar > 0.05:
                        # Charge toward high SOC using solar
                        rate = np.clip((self.soc_target_solar_charge - soc_cool) * 1.0, 0.0, self.max_storage_rate)
                        a.append(rate)
                    elif self._is_peak_hour(hour):
                        # Discharge during peaks
                        rate = -np.clip((soc_cool - self.soc_target_peak_discharge) * 1.0, 0.0, self.max_storage_rate)
                        a.append(rate)
                    else:
                        # Drift back to mid SOC
                        delta = (self.soc_target_idle - soc_cool)
                        rate = np.clip(delta, -self.max_storage_rate, self.max_storage_rate)
                        a.append(rate)
                elif a_name == 'dhw_storage':
                    if solar > 0.05:
                        rate = np.clip((self.soc_target_solar_charge - soc_dhw) * 1.0, 0.0, self.max_storage_rate)
                        a.append(rate)
                    elif self._is_peak_hour(hour):
                        rate = -np.clip((soc_dhw - self.soc_target_peak_discharge) * 1.0, 0.0, self.max_storage_rate)
                        a.append(rate)
                    else:
                        delta = (self.soc_target_idle - soc_dhw)
                        rate = np.clip(delta, -self.max_storage_rate, self.max_storage_rate)
                        a.append(rate)
                else:
                    # Unknown action, hold
                    a.append(0.0)

            actions.append(a)

        return actions


# Optional: a more principled storage MPC using cvxpy (single-step control, 24h preview)
try:
    import cvxpy as cp
except Exception:
    cp = None

class StorageMPC(BasicRBCAgent):
    """
    Convex storage MPC with 24-step horizon:
    - Controls cooling_storage and dhw_storage flows within [-r_max, r_max]
    - Tracks SOC targets: charge on solar hours, discharge on peak hours, hold otherwise
    - Cooling device remains proportional for simplicity

    Note: This is a simplified convex MPC, not using full building thermal dynamics.
    """
    def __init__(self, env: 'local_evaluation.WrapperEnv', horizon: int = 24):
        super().__init__(env)
        self.horizon = horizon
        self.comfort_band = 0.5
        self.cooling_gain = 4.0
        self.r_max = 0.3
        self.soc_min = 0.05
        self.soc_max = 0.95
        self.soc_idle = 0.6
        self.soc_target_solar = 0.9
        self.soc_target_peak = 0.3

    def _is_peak_hour(self, hour: float) -> bool:
        return (6 <= hour <= 9) or (18 <= hour <= 22)

    def _target_soc(self, hour: float, solar: float) -> float:
        if solar > 0.05:
            return self.soc_target_solar
        if self._is_peak_hour(hour):
            return self.soc_target_peak
        return self.soc_idle

    def predict(self, observations: 'list[list[float]]') -> 'list[list[float]]':
        if cp is None:
            # Fallback to SimpleMPC behavior if cvxpy not available
            return SimpleMPC(self).predict(observations)

        actions = []
        for action_names, obs_names, obs in zip(self.action_names, self.observation_names, observations):
            hour = obs[obs_names.index('hour')] if 'hour' in obs_names else 12.0
            Tin = obs[obs_names.index('indoor_dry_bulb_temperature')]
            if 'indoor_dry_bulb_temperature_set_point' in obs_names:
                Tsp = obs[obs_names.index('indoor_dry_bulb_temperature_set_point')]
            elif 'indoor_dry_bulb_temperature_cooling_set_point' in obs_names:
                Tsp = obs[obs_names.index('indoor_dry_bulb_temperature_cooling_set_point')]
            else:
                Tsp = 22.0
            solar = obs[obs_names.index('solar_generation')] if 'solar_generation' in obs_names else 0.0
            soc_cool = obs[obs_names.index('cooling_storage_soc')] if 'cooling_storage_soc' in obs_names else 0.5
            soc_dhw = obs[obs_names.index('dhw_storage_soc')] if 'dhw_storage_soc' in obs_names else 0.5

            # Build simple 24h preview signals from current hour and solar persistence
            hours = np.array([(hour + k) % 24 for k in range(self.horizon)], dtype=float)
            solars = np.full(self.horizon, solar)
            soc_targets_cool = np.array([self._target_soc(h, s) for h, s in zip(hours, solars)])
            soc_targets_dhw = soc_targets_cool.copy()

            # Decision variables: charge(+)/discharge(-) rates for each storage
            u_cool = cp.Variable(self.horizon)
            u_dhw = cp.Variable(self.horizon)

            # SOC trajectories
            soc_c = soc_cool + cp.cumsum(u_cool)
            soc_d = soc_dhw + cp.cumsum(u_dhw)

            constraints = []
            constraints += [cp.abs(u_cool) <= self.r_max, cp.abs(u_dhw) <= self.r_max]
            constraints += [soc_c >= self.soc_min, soc_c <= self.soc_max]
            constraints += [soc_d >= self.soc_min, soc_d <= self.soc_max]

            # Objective: track targets and penalize moves
            obj = cp.sum_squares(soc_c - soc_targets_cool) + cp.sum_squares(soc_d - soc_targets_dhw)
            obj += 0.1 * (cp.sum_squares(u_cool) + cp.sum_squares(u_dhw))

            prob = cp.Problem(cp.Minimize(obj), constraints)
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception:
                # Fallback solve
                prob.solve(warm_start=True, verbose=False)

            u0_cool = float(np.clip(u_cool.value[0] if u_cool.value is not None else 0.0, -self.r_max, self.r_max))
            u0_dhw = float(np.clip(u_dhw.value[0] if u_dhw.value is not None else 0.0, -self.r_max, self.r_max))

            # Cooling device proportional
            violation = Tin - Tsp - self.comfort_band
            cool_power = 0.0 if violation <= 0 else min(1.0, violation / self.cooling_gain)

            a = []
            for a_name in action_names:
                if a_name == 'cooling_device':
                    a.append(cool_power)
                elif a_name == 'cooling_storage':
                    a.append(u0_cool)
                elif a_name == 'dhw_storage':
                    a.append(u0_dhw)
                else:
                    a.append(0.0)

            actions.append(a)

        return actions


# Integration of mpc_benchmark-main's BuildingEnergyMPC into a CityLearn agent
class BenchmarkMPC(BasicRBCAgent):
    """
    Wraps mpc_benchmark-main's BuildingEnergyMPC to act as a CityLearn agent.
    Uses per-step persistence forecasts for weather and occupancy.
    Controls: cooling_device, cooling_storage, dhw_storage.
    """
    def __init__(self, env: 'local_evaluation.WrapperEnv', horizon: int = 12, control_horizon: int = 3):
        super().__init__(env)
        import os, sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mpc_dir = os.path.join(here, 'mpc_benchmark-main')
        if mpc_dir not in sys.path:
            sys.path.insert(0, mpc_dir)
        try:
            from building_mpc import BuildingEnergyMPC
        except Exception as e:
            raise RuntimeError(f"Failed to import BuildingEnergyMPC: {e}")
        # One MPC controller per building
        self._mpc_by_building = []
        for _ in range(len(self.action_names)):
            self._mpc_by_building.append(
                BuildingEnergyMPC(
                    prediction_horizon=horizon,
                    control_horizon=control_horizon,
                    sampling_time=1.0,
                    comfort_temp_range=(20.0, 26.0),
                    comfort_humidity_range=(35.0, 70.0),
                    use_saudi_pricing=False,
                    penalty_comfort=200.0,
                    penalty_control_effort=0.1,
                    hvac_efficiency=3.2,
                    include_storage=True,
                    safety_factor=1.1,
                )
            )
        # Track previous control per building
        self._prev_controls = [
            {
                'prev_T_set': 22.0,
                'prev_RH_set': 50.0,
                'prev_cooling_storage': 0.0,
                'prev_dhw_storage': 0.0,
                'prev_cooling_device': 0.0,
            }
            for _ in range(len(self.action_names))
        ]
        # Planned sequences to reuse without solving every step
        self._planned_sequences = [[] for _ in range(len(self.action_names))]

    def _build_initial_state(self, obs_names, obs, prev):
        state = {
            'temperature': float(obs[obs_names.index('indoor_dry_bulb_temperature')]),
            'humidity': float(obs[obs_names.index('indoor_relative_humidity')]) if 'indoor_relative_humidity' in obs_names else 50.0,
            'mass_temperature': float(obs[obs_names.index('indoor_dry_bulb_temperature')]),
            'prev_T_set': prev['prev_T_set'],
            'prev_RH_set': prev['prev_RH_set'],
            'cooling_storage_soc': float(obs[obs_names.index('cooling_storage_soc')]) if 'cooling_storage_soc' in obs_names else 0.5,
            'dhw_storage_soc': float(obs[obs_names.index('dhw_storage_soc')]) if 'dhw_storage_soc' in obs_names else 0.5,
            'prev_cooling_storage': prev['prev_cooling_storage'],
            'prev_dhw_storage': prev['prev_dhw_storage'],
            'prev_cooling_device': prev['prev_cooling_device'],
        }
        return state

    def _persistence_forecasts(self, obs_names, obs, horizon):
        hour = float(obs[obs_names.index('hour')]) if 'hour' in obs_names else 12.0
        Tout = float(obs[obs_names.index('outdoor_dry_bulb_temperature')]) if 'outdoor_dry_bulb_temperature' in obs_names else 35.0
        RHout = float(obs[obs_names.index('outdoor_relative_humidity')]) if 'outdoor_relative_humidity' in obs_names else 30.0
        solar = float(obs[obs_names.index('solar_generation')]) if 'solar_generation' in obs_names else 0.0
        weather_forecast = [{
            'temperature': Tout,
            'humidity': RHout,
            'solar_radiation': solar,
        } for _ in range(horizon)]
        occupancy_forecast = [0.0 for _ in range(horizon)]
        building_demands_forecast = [{
            'cooling_demand': 0.0,
            'dhw_demand': 0.0,
            'non_shiftable_load': 0.0,
        } for _ in range(horizon)]
        return weather_forecast, occupancy_forecast, building_demands_forecast

    def predict(self, observations: 'list[list[float]]') -> 'list[list[float]]':
        actions_all = []
        for b_idx, (action_names, obs_names, obs) in enumerate(zip(self.action_names, self.observation_names, observations)):
            # If we have a planned sequence, use next planned control
            if self._planned_sequences[b_idx]:
                planned = self._planned_sequences[b_idx].pop(0)
                a = []
                for a_name in action_names:
                    a.append(planned.get(a_name, 0.0))
                actions_all.append(a)
                # Update prev with what we applied
                self._prev_controls[b_idx]['prev_cooling_storage'] = planned.get('cooling_storage', 0.0)
                self._prev_controls[b_idx]['prev_dhw_storage'] = planned.get('dhw_storage', 0.0)
                self._prev_controls[b_idx]['prev_cooling_device'] = planned.get('cooling_device', 0.0)
                continue

            mpc = self._mpc_by_building[b_idx]
            prev = self._prev_controls[b_idx]
            state = self._build_initial_state(obs_names, obs, prev)
            weather_fc, occ_fc, dem_fc = self._persistence_forecasts(obs_names, obs, mpc.N_pred)
            try:
                optimal, info = mpc.solve_mpc(state, weather_fc, occ_fc, dem_fc)
            except Exception:
                optimal = {
                    'cooling_storage_charge': 0.0,
                    'dhw_storage_charge': 0.0,
                    'cooling_device_intensity': 0.0,
                }
                info = {'success': False}

            # Map current action
            a = []
            mapped_current = {}
            for a_name in action_names:
                if a_name == 'cooling_device':
                    val = float(np.clip(optimal.get('cooling_device_intensity', 0.0), 0.0, 1.0))
                    a.append(val)
                    mapped_current['cooling_device'] = val
                elif a_name == 'cooling_storage':
                    val = float(np.clip(optimal.get('cooling_storage_charge', 0.0), -1.0, 1.0))
                    a.append(val)
                    mapped_current['cooling_storage'] = val
                elif a_name == 'dhw_storage':
                    val = float(np.clip(optimal.get('dhw_storage_charge', 0.0), -1.0, 1.0))
                    a.append(val)
                    mapped_current['dhw_storage'] = val
                else:
                    a.append(0.0)
            actions_all.append(a)
            # Update prev
            prev['prev_cooling_storage'] = mapped_current.get('cooling_storage', 0.0)
            prev['prev_dhw_storage'] = mapped_current.get('dhw_storage', 0.0)
            prev['prev_cooling_device'] = mapped_current.get('cooling_device', 0.0)

            # If solver returned a sequence, plan ahead to avoid frequent solves
            seq = []
            opt_seq = info.get('optimal_sequence') if isinstance(info, dict) else None
            if opt_seq is not None:
                # Build mapped plans for remaining control horizon-1 steps
                for row in opt_seq[1:]:
                    plan = {
                        'cooling_device': float(np.clip(row[4] if len(row) > 4 else 0.0, 0.0, 1.0)),
                        'cooling_storage': float(np.clip(row[2] if len(row) > 2 else 0.0, -1.0, 1.0)),
                        'dhw_storage': float(np.clip(row[3] if len(row) > 3 else 0.0, -1.0, 1.0)),
                    }
                    seq.append(plan)
            self._planned_sequences[b_idx].extend(seq)

        return actions_all 