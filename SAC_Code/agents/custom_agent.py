import os
from pathlib import Path
from typing import List, Union
from citylearn.citylearn import CityLearnEnv
# from citylearn.utilities import read_json
from citylearn.wrappers import StableBaselines3Wrapper
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3.sac import SAC
from agents.rbc_agent import BasicRBCAgent
import torch
import json

# Local JSON reader to replace deprecated/removed citylearn.utilities.read_json

def read_json(filepath: Union[str, Path]):
    with open(filepath, 'r') as file:
        return json.load(file)

class RBC(BasicRBCAgent):
    """
    Custom RBC agent that controls `cooling_device` like a single-stage heat pump only turning on
    when the temperature falls outside a specified comfort band about the setpoint. Meanwhile,
    storage systems are only charged when there is solar generation and discharged within the 
    morning (6 - 9 AM) and evening (6 - 10 PM) peak periods.

    Notes
    -----
    Make sure to inherit from the :py:class:`agents.rbc_agent.BasicRBCAgent` class provided by AICrowd
    and calls this super class' `__init__` method during initialization to make sure your agent is s
    etup correctly to interface with their online evaluator.

    Also, note that the `env` used to initialize this class is not the CityLearn environment type
    (:py:class:`citylearn.citylearn.CityLearnEnv`). Instead it is a wrapper for it provided by
    AICrowd (:py:class:`local_evaluation.WrapperEnv`) that only provides information about static
    environment metadata e.g. observation and action names.
    """
    
    def __init__(self, env: 'local_evaluation.WrapperEnv'):
        super().__init__(env)
        self.comfort_band = 1.0

    def predict(self, observations: List[List[float]]) -> List[List[float]]:
        actions = []

        for a, n, o in zip(self.action_names, self.observation_names, observations):
            hour = o[n.index('hour')]
            indoor_temperature = o[n.index('indoor_dry_bulb_temperature')]
            # Handle both possible setpoint observation names
            if 'indoor_dry_bulb_temperature_set_point' in n:
                setpoint = o[n.index('indoor_dry_bulb_temperature_set_point')]
            elif 'indoor_dry_bulb_temperature_cooling_set_point' in n:
                setpoint = o[n.index('indoor_dry_bulb_temperature_cooling_set_point')]
            else:
                setpoint = 22.0  # default setpoint
            temperature_difference = indoor_temperature - setpoint
            solar_generation = o[n.index('solar_generation')] if 'solar_generation' in n else 0.0
            actions_ = []

            for a_n in a:
                # Run the cooling device as if it were a single-stage heat pump;
                # when the indoor temperature rises above a defined comfort band,
                # the heat pump should come on otherwise, it is off. Using 80% 
                # arbitrarily but 100% can also be used. However, the heat pump in
                # CityLearn is set up to allow continuous power control so any action
                # between 0 and 100% is possible. Knowing, this, and how the COP of
                # a heat pump changes w.r.t. to outdoor temperature, you can edit these
                # rules by discretizing the actions further.
                if a_n == 'cooling_device':
                    if temperature_difference > self.comfort_band:
                        actions_.append(0.8)
                    
                    else:
                        actions_.append(0.0)
                
                # The remaining actions are for the energy storage systems:
                # electrical storage and DHW storage. For these, we will charge them
                # by 10% each hour when there is solar generation. Then, during
                # the morning and evening peaks, we will discharge. Other times,
                # the storage systems are not acted on. You can tailor these rules to
                # better target the exact hours when peaks occur or to discharge when
                # ramping is least.
                elif solar_generation > 0.0:
                    actions_.append(0.15)

                else:
                    if 6 <= hour <= 9 or 18 <= hour <= 21:
                        actions_.append(-0.15)

                    else:
                        actions_.append(0.0)
                
            actions.append(actions_)
        
        return actions

class SB3SAC(BasicRBCAgent):
    """
    Wrapper for training and evaluating on Stable Baselines3 SAC reinforcement learning agent. 
    Assumes :py:class:`citylearn.citylearn.CityLearnEnv.central_agent`=`True`.

    Call the initialization function for evaluation (inference) only. To train the agent, use the 
    :py:meth:`agents.custom_agent.SB3SAC.learn` classmethod. The trained agent is saved to the 
    filepath defined by :py:const:`agents.custom_agent.SB3SAC.MODEL_FILEPATH`.

    Notes
    -----
    Make sure to inherit from the :py:class:`agents.rbc_agent.BasicRBCAgent` class provided by AICrowd
    and calls this super class' `__init__` method during initialization to make sure your agent is s
    etup correctly to interface with their online evaluator.

    Also, note that the `env` used to initialize this class is not the CityLearn environment type
    (:py:class:`citylearn.citylearn.CityLearnEnv`). Instead it is a wrapper for it provided by
    AICrowd (:py:class:`local_evaluation.WrapperEnv`) that only provides information about static
    environment metadata e.g. observation and action names.
    """
    
    # You can change the filepath to somewhere else or another name but make sure it is saved within
    # the starter kit so that when you make a submission, it is upload with your submission and the
    # online evaluator is able to find it during inference.
    MODEL_FILEPATH = os.path.join('trained_models', 'sb3sac')

    def __init__(self, env: 'local_evaluation.WrapperEnv'):
        super().__init__(env)
        assert len(self.observation_names) == 1, ('SB3SAC only compatible with'
            ' citylearn.citylearn.CityLearnEnv.central_agent = True or 1-building environment.')
        self.model = SAC.load(self.MODEL_FILEPATH) # load saved trained model

        # Get list of active observations so that observations parsed to agent when selecting actions
        # match those used during training.
        self.active_observations = read_json('aicrowd.json')['active_observations']

        # 
        self.building_observation_indices = self.__set_building_observation_indices()

    def predict(self, observations):
        """Select deterministic actions for each building then combined into one list. Pads observations if needed for compatibility."""

        actions = []
        observations = observations[0]

        # Determine expected observation size from model 3
        expected_obs_size = self.model.observation_space.shape[0] # helper for more than 7 observations

        for ixs in self.building_observation_indices:
            o = np.array([observations[ix] for ix in ixs], dtype=float)
            # Pad or truncate observation as needed3
            if len(o) < expected_obs_size: # 3
                padded_o = np.zeros(expected_obs_size) # helper for more than 7 observations
                padded_o[:len(o)] = o # helper for more than 7 observations
                o = padded_o # helper for more than 7 observations
            elif len(o) > expected_obs_size: # helper for more than 7 observations
                o = o[:expected_obs_size] # helper for more than 7 observations
            actions_, _ = self.model.predict(o, deterministic=True)
            # Ensure we only get the expected number of actions per building (3) 3 helper for more than 7 observations
            if len(actions_) > 3: # 3 helper for more than 7 observations
                actions_ = actions_[:3] # 3 helper for more than 7 observations
            actions.extend(actions_)

        return [actions] 

    @classmethod
    def learn(
        cls, schema: Union[Path, str], episodes_per_building: int, env_kwargs: dict = None, 
        model_kwargs: dict = None, learn_kwargs: dict = None
    ):
        """Train Stable-Baselines3 SAC agent using CityLearn `schema`. Only one policy is trained for all buildings 
        by switching the building used for training after an episode completes. The cycle is repeated for 
        `episodes_per_building` episodes. This ensures that the policy's network structure is not constrained to a fixed 
        number of buildings and can generalize to any number of buildings used in the online evaluation. Use `env_kwargs` to
        parse custom initialization parameters for :py:class:`citylearn.citylearn.CityLearnEnv`. Use `model_kwargs` to parse
        custom initialization parameters for :py:class:`stable_baselines3.sac.SAC` and `learn_kwargs` to parse parameters to
        :py:meth:`stable_baselines3.sac.SAC.learn`."""

        # set defaults for env and agent initialization 
        env_kwargs = {} if env_kwargs is None else env_kwargs
        model_kwargs = {} if model_kwargs is None else model_kwargs
        learn_kwargs = {} if learn_kwargs is None else learn_kwargs
        model = None
        
        # Check if GPU is available and configure device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Add device to model_kwargs if not already specified
        if 'device' not in model_kwargs:
            model_kwargs['device'] = device
        
        # To train the agent on one building at a time,  we need to know the number of building
        # in the environment. we can infer that from the schema
        schema_filepath = schema # to keep record of the orignal schema filepath that was parsed
        schema = read_json(schema)
        building_count = len(schema['buildings'])

        # Since we will parse in the schema as a dictionary,we also need to let the schema know 
        # the root directory to find all files used to construct the environment
        schema['root_directory'] = os.path.split(Path(schema_filepath).absolute())[0]        

        # Next, we want to be able to tailor the observation space to only observations relevant 
        # to the object we want to optimize. For example a control problem that only focuses on 
        # comfort, adding one-too-many observations that are unrelated to this objective and are 
        # not driven by the agent actions could make the observation space noisy and learning difficult. 
        # So in a comfort problem, we might only want to set temperature and temporal related observations 
        # active. In other problem formulations, more or fewer observations may be need. To set the active 
        # observations, we will read them in from aicrowd.json with the key 'active_observations' if 
        # specified and override the the observations already set to be active in the schema otherwise, 
        # we keep the observations as provided in the schema. If we will loop through all the observations 
        # in the schema; if an observation is in the list of active observations, we will set its 'active' 
        # value to True, otherwise False.
        active_observations = read_json('aicrowd.json').get('active_observations')
        
        if active_observations is not None and len(active_observations) > 0:
            for o in schema['observations']:
                if o in active_observations:
                    schema['observations'][o]['active'] = True

                else:
                    schema['observations'][o]['active'] = False
        
        else:
            pass
        
        # The agent needs to know the range of observation and action values to expect to be able to carry out 
        # random sampling carry out internal normalizations if any, and have a sense of what observation and 
        # action values are valid. The way it gets this information is through an gym.Spaces instance for either 
        # observations or actions. In CityLearn, observations and actions are continuous thus, are defined using 
        # the gym.Spaces.Box type which is suitable for continuous spaces. However, we want to train our agent 
        # in such a way that during evaluation (inference), if the public dataset has three buildings but the 
        # evaluation dataset has more, say 6, we are not constrained by the structure of the agent's neural 
        # network where it is expecting three buildings' worth or observations to predict three buildings' worth 
        # of actions. Instead, we will design the agent to take in 1 building's worth of observations and predict 
        # one building's worth of actions and then, we can loop through whatever number of buildings independent 
        # observations and select actions for each building. To make this possible, we will train one building at 
        # a time making it possible to have a generalizable agent neural network structure. The caveat is that the 
        # agent during training needs to know what the observation and action spaces are for the environment it is 
        # training. Each building is unique causing unique space limit estimates and if the space definition changes 
        # from what was set initially for the agent, it will throw an error. To get around this, we will impose the 
        # same space limits on all the buildings by setting the high and low limits from all buildings limits.
        # We will first need to get the limit estimates made internally in CityLearn.
        env = CityLearnEnv(schema, central_agent=False)
        observation_space = [s.low for s in env.observation_space] + [s.high for s in env.observation_space]
        action_space = [s.low for s in env.action_space] + [s.high for s in env.action_space]

        # Next we set general observation and action spaces within the limits of all buildings at the same time.
        observation_space = Box(
            low=np.array(observation_space).min(axis=0), 
            high=np.array(observation_space).max(axis=0), 
            dtype='float32'
        )
        action_space = Box(
            low=np.array(action_space).min(axis=0), 
            high=np.array(action_space).max(axis=0), 
            dtype='float32'
        )
        
        # Now we are ready to start training. We will the agent on each building in sequence.
        # The sequence will be repeated for `episodes_per_building` episodes.
        for e in range(episodes_per_building):
            for b in range(building_count):
                print('B:',b , 'E:', e)
                env_kwargs['buildings'] = [b] # set current building to train on
                env_kwargs['central_agent'] = True # central agent must be True to work with SB3
                env = CityLearnEnv(schema, **env_kwargs) # initialize environment
                env.buildings[0].observation_space = observation_space # set general observation space
                env.buildings[0].action_space = action_space # set general action space
                episode_steps = env.time_steps  # capture episode length before wrapping
                env = StableBaselines3Wrapper(env) # a wrapper to establish CityLearn <-> SB3 interface

                # By the first building in the first train sequence, the model has not been initialized.
                # So, we initialize the model this one time and subsequent training iterations, we will
                # only update the environment it trains on
                if model is None:
                    print(f"🎯 Initializing SAC model on {device.upper()}")
                    model = SAC('MlpPolicy', env, **model_kwargs)
                
                else:
                    model.set_env(env)

                # Here we set how long we want to train the current environment (or building) for.
                # We have a finite episode so we will just set it to the length of the episode.
                learn_kwargs['total_timesteps'] = episode_steps
                learn_kwargs['reset_num_timesteps'] = False
                model.learn(**learn_kwargs)
        
        # Finally, save the trained agent to disk so that it can be used for inference later on.
        model.save(cls.MODEL_FILEPATH)
        print(f"💾 Model saved to {cls.MODEL_FILEPATH}")

    def __set_building_observation_indices(self):
        """Get the index positions for each building's observations in the observation 
        list that will be parsed to the predict function."""

        observation_names = self.observation_names[0]
        observation_indices = []

        for i, o in enumerate(observation_names):
            if o in self.active_observations and o not in observation_names[:i]:
                observation_indices.append([i for i, v in enumerate(observation_names) if v == o])
            
            else:
                pass

        building_observation_indices = [[] for _ in range(len(self.building_metadata))]

        for ixs in observation_indices:
            if len(ixs) == 1:
                for b in building_observation_indices:
                    b.append(ixs[0])

            else:
                for b, ix in zip(building_observation_indices, ixs):
                    b.append(ix)

        return building_observation_indices
