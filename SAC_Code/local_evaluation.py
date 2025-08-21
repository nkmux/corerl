import numpy as np
import time
import os
import json
import math

from citylearn.citylearn import CityLearnEnv
from citylearn.dynamics import LSTMDynamics

"""
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.user_agent import SubmissionAgent
from rewards.user_reward import SubmissionReward

class LSTMDynamicsAugmentor:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        # resolve root directory
        root = self.schema.get('root_directory')
        if root is None:
            root = os.path.dirname(os.path.abspath(schema_path))
        self.root = root
        # build per-building models and buffers
        self.models = []
        self.buffers = []  # one buffer per building: list of normalized vectors history
        self.input_specs = []  # tuples (names, mins, maxs, lookback)
        for b_name in self.schema['buildings']:
            d = self.schema['buildings'][b_name].get('dynamics')
            if d is None:
                self.models.append(None)
                self.buffers.append(None)
                self.input_specs.append(None)
                continue
            attrs = d.get('attributes', {})
            # map filename to filepath
            filename = attrs.get('filename')
            if filename:
                attrs = dict(attrs)
                attrs['filepath'] = os.path.join(self.root, filename)
                attrs.pop('filename', None)
            model = LSTMDynamics(**attrs)
            model.reset()
            self.models.append(model)
            self.buffers.append([])
            self.input_specs.append((model.input_observation_names, model.input_normalization_minimum, model.input_normalization_maximum, model.lookback))

    @staticmethod
    def _safe_index(names, key, default=None):
        return names.index(key) if key in names else default

    @staticmethod
    def _periodic(val, period):
        angle = 2.0 * math.pi * (float(val) % period) / period
        return math.sin(angle), math.cos(angle)

    @staticmethod
    def _minmax(x, xmin, xmax):
        if xmin is None or xmax is None or xmax == xmin:
            return float(x)
        return float((x - xmin) / (xmax - xmin))

    def apply(self, observations, observation_names):
        # observations: List[List[float]], observation_names: List[List[str]]
        new_obs = []
        for bi, (o, names) in enumerate(zip(observations, observation_names)):
            model = self.models[bi]
            if model is None:
                new_obs.append(o)
                continue
            input_names, mins, maxs, lookback = self.input_specs[bi]

            # Build raw input vector in specified order
            month = o[self._safe_index(names, 'month', 0)]
            hour = o[self._safe_index(names, 'hour', 0)]
            day_type = o[self._safe_index(names, 'day_type', 1)]
            m_sin, m_cos = self._periodic(month if month else 1, 12)
            h_sin, h_cos = self._periodic(hour if hour else 0, 24)
            # Map holiday (8) to 1 for periodicity
            dt = 1 if int(day_type) == 8 else int(day_type)
            dt_sin, dt_cos = self._periodic(dt, 7)

            lookup = {
                'direct_solar_irradiance': o[self._safe_index(names, 'direct_solar_irradiance', 0)] if 'direct_solar_irradiance' in names else 0.0,
                'diffuse_solar_irradiance': o[self._safe_index(names, 'diffuse_solar_irradiance', 0)] if 'diffuse_solar_irradiance' in names else 0.0,
                'outdoor_dry_bulb_temperature': o[self._safe_index(names, 'outdoor_dry_bulb_temperature', 0)] if 'outdoor_dry_bulb_temperature' in names else 0.0,
                'indoor_dry_bulb_temperature_cooling_set_point': o[self._safe_index(names, 'indoor_dry_bulb_temperature_cooling_set_point', None)] if 'indoor_dry_bulb_temperature_cooling_set_point' in names else (o[self._safe_index(names, 'indoor_dry_bulb_temperature_set_point', None)] if 'indoor_dry_bulb_temperature_set_point' in names else 22.0),
                'occupant_count': o[self._safe_index(names, 'occupant_count', 0)] if 'occupant_count' in names else 0.0,
                'cooling_demand': o[self._safe_index(names, 'cooling_demand', 0)] if 'cooling_demand' in names else 0.0,
                'heating_demand': o[self._safe_index(names, 'heating_demand', 0)] if 'heating_demand' in names else 0.0,
                'month_sin': m_sin,
                'month_cos': m_cos,
                'hour_sin': h_sin,
                'hour_cos': h_cos,
                'day_type_sin': dt_sin,
                'day_type_cos': dt_cos,
                'indoor_dry_bulb_temperature': o[self._safe_index(names, 'indoor_dry_bulb_temperature')]
            }

            raw_vec = []
            for key in input_names:
                raw_vec.append(lookup.get(key, 0.0))

            # Normalize
            norm_vec = [self._minmax(v, mins[i], maxs[i]) for i, v in enumerate(raw_vec)]

            # Manage buffer of length lookback+1
            buf = self.buffers[bi]
            if len(buf) == 0:
                # prefill
                for _ in range(lookback + 1):
                    buf.append(norm_vec)
            else:
                buf.append(norm_vec)
                if len(buf) > (lookback + 1):
                    buf.pop(0)

            # Prepare tensor input shape (1, lookback+1, input_size)
            x = np.array(buf, dtype=np.float32)
            x = x.reshape(1, x.shape[0], x.shape[1])
            # forward
            import torch
            x_t = torch.from_numpy(x)
            y, h = model.forward(x_t, model._hidden_state)
            model._hidden_state = h
            pred = float(y.detach().cpu().numpy().squeeze())

            # Replace indoor temperature in obs
            o_new = list(o)
            idx_t = names.index('indoor_dry_bulb_temperature') if 'indoor_dry_bulb_temperature' in names else None
            if idx_t is not None:
                o_new[idx_t] = pred
            new_obs.append(o_new)

        return new_obs

class WrapperEnv:
    """
    Env to wrap provide Citylearn Env data without providing full env
    Preventing attribute access outside of the available functions
    """
    def __init__(self, env_data):
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.observation_space = env_data['observation_space']
        self.action_space = env_data['action_space']
        self.time_steps = env_data['time_steps']
        self.seconds_per_time_step = env_data['seconds_per_time_step']
        self.random_seed = env_data['random_seed']
        self.buildings_metadata = env_data['buildings_metadata']
        self.episode_tracker = env_data['episode_tracker']
    
    @property
    def unwrapped(self):
        """Return self to make compatible with CityLearn agents that expect unwrapped attribute."""
        return self
    
    def get_metadata(self):
        return {'buildings': self.buildings_metadata}

def patch_building_for_evaluation(env):
    """
    Patch the missing attributes that cause evaluation bugs in CityLearn 2.2.0
    """
    for building in env.buildings:
        # Patch missing net_electricity_consumption variants
        if not hasattr(building, 'net_electricity_consumption_without_storage_and_partial_load'):
            building.net_electricity_consumption_without_storage_and_partial_load = building.net_electricity_consumption.copy()
        
        if not hasattr(building, 'net_electricity_consumption_emission_without_storage_and_partial_load'):
            building.net_electricity_consumption_emission_without_storage_and_partial_load = building.net_electricity_consumption.copy()
            
        if not hasattr(building, 'net_electricity_consumption_cost_without_storage_and_partial_load'):
            building.net_electricity_consumption_cost_without_storage_and_partial_load = building.net_electricity_consumption.copy()
            
        # Add any other missing attributes that might be needed
        if not hasattr(building, 'cooling_electricity_consumption_without_storage_and_partial_load'):
            if hasattr(building, 'cooling_electricity_consumption'):
                building.cooling_electricity_consumption_without_storage_and_partial_load = building.cooling_electricity_consumption.copy()
            else:
                building.cooling_electricity_consumption_without_storage_and_partial_load = building.net_electricity_consumption.copy() * 0.6  # rough estimate
                
        if not hasattr(building, 'dhw_electricity_consumption_without_storage_and_partial_load'):
            if hasattr(building, 'dhw_electricity_consumption'):
                building.dhw_electricity_consumption_without_storage_and_partial_load = building.dhw_electricity_consumption.copy()
            else:
                building.dhw_electricity_consumption_without_storage_and_partial_load = building.net_electricity_consumption.copy() * 0.2  # rough estimate
    
    return env

def create_citylearn_env(config, reward_function, central_agent: bool = True):  # Changed default to True
    env = CityLearnEnv(config.SCHEMA, reward_function=reward_function, central_agent=central_agent)

    env_data = dict(
        observation_names = env.observation_names,
        action_names = env.action_names,
        observation_space = env.observation_space,
        action_space = env.action_space,
        time_steps = env.time_steps,
        random_seed = None,
        episode_tracker = None,
        seconds_per_time_step = None,
        buildings_metadata = env.get_metadata()['buildings']
    )

    wrapper_env = WrapperEnv(env_data)
    return env, wrapper_env

def update_power_outage_random_seed(env: CityLearnEnv, random_seed: int) -> CityLearnEnv:
    """Update random seed used in generating power outage signals.
    
    Used to optionally update random seed for stochastic power outage model in all buildings.
    Random seeds should be updated before calling :py:meth:`citylearn.citylearn.CityLearnEnv.reset`.
    """

    for b in env.buildings:
        if hasattr(b, 'stochastic_power_outage_model'):
            b.stochastic_power_outage_model.random_seed = random_seed

    return env

def evaluate(config):
    print("Starting local evaluation")
    
    # Use central_agent=True for compatibility with trained SAC model
    env, wrapper_env = create_citylearn_env(config, SubmissionReward, central_agent=True)
    print("Env Created")

    agent = SubmissionAgent(wrapper_env)
    dynamics = LSTMDynamicsAugmentor(config.SCHEMA)

    # Handle both tuple and list return formats from reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        observations = reset_result[0]
    else:
        observations = reset_result

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(observations)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        while True:
            
            ### This is only a reference script provided to allow you 
            ### to do local evaluation. The evaluator **DOES NOT** 
            ### use this script for orchestrating the evaluations. 

            step_result = env.step(actions)
            
            # Handle different step return formats (Gymnasium vs Gym)
            if len(step_result) == 4:
                observations, reward, done, info = step_result
            else:
                observations, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                
            if not done:
                step_start = time.perf_counter()
                # apply dynamics before predicting actions
                observations = dynamics.apply(observations, wrapper_env.observation_names)
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                episodes_completed += 1
                
                # Get official metrics with patching
                try:
                    env = patch_building_for_evaluation(env)
                    metrics_df = env.evaluate_citylearn_challenge()
                    episode_metrics.append(metrics_df)
                    # Format metrics for readability
                    try:
                        df = metrics_df
                        # filter to district-level rows if available
                        if hasattr(df, 'columns') and 'level' in df.columns:
                            df = df[df['level'] == 'district']
                        # build summary dict
                        summary = {}
                        weights = []
                        weighted = []
                        for _, row in df.iterrows():
                            name = row.get('cost_function', None)
                            value = float(row.get('value', float('nan')))
                            weight = row.get('weight', None)
                            if name is None:
                                continue
                            summary[name] = {'display_name': name, 'weight': weight, 'value': value}
                            if weight is not None and not np.isnan(value):
                                weights.append(float(weight))
                                weighted.append(float(weight) * value)
                        if len(weighted) > 0:
                            summary['average_score'] = {'display_name': 'Score', 'weight': None, 'value': float(np.nansum(weighted))}
                        print(f"Episode complete: {episodes_completed} | Latest episode metrics: {summary}")
                    except Exception:
                        # fallback to plain print
                        print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics_df}")
                except Exception as e:
                    print(f"Episode complete: {episodes_completed} | Metrics evaluation failed: {e}")
                
                # Update power outage random seed for different episodes
                env = update_power_outage_random_seed(env, 90000 + episodes_completed)

                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    observations = reset_result[0]
                else:
                    observations = reset_result

                step_start = time.perf_counter()
                observations = dynamics.apply(observations, wrapper_env.observation_names)
                actions = agent.predict(observations)
                agent_time_elapsed += time.perf_counter()- step_start
            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= config.num_episodes:
                break

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    print(f"Total time taken by agent: {agent_time_elapsed}s")
    

if __name__ == '__main__':
    class Config:
        data_dir = './data_saudi/' #put data_saudi 
        SCHEMA = os.path.join(data_dir, 'schema.json')  # Use upgraded original schema
        num_episodes = 15  # Run multiple episodes for better evaluation
    
    config = Config()

    evaluate(config)
