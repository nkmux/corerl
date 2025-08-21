import os
from agents.custom_agent import SB3SAC
from rewards.custom_reward import ComfortandConsumptionReductionReward

def main():
    """
    Use to train a Stable Baseline3 SAC agent. Can be adapted for other Stable Baselines3 agents.
    
    Only one policy is trained for all buildings by switching the building used for training after 
    an episode completes. The cycle is repeated for `episodes_per_building` episodes. This ensures 
    that the policy's network structure is not constrained to a fixed number of buildings and can 
    generalize to any number of buildings used in the online evaluation.

    To run this script, execute `python train_sb3sac.py` from the terminal.
    """

    # The CityLearn dataset to use for training the agent (use upgraded original schema)
    schema = os.path.join('data_saudi', 'schema.json')

    # Use the env_kwargs to specify any custom values that override defaults for parameters used to 
    # initialize CityLearn (see https://github.com/intelligent-environments-lab/CityLearn/blob/8b3433afd59130b81aa3e6d56e449ba1184035ec/citylearn/citylearn.py#L94)
    # Here, we are setting the reward_function to our custom defined reward function
    env_kwargs = {
        'reward_function': ComfortandConsumptionReductionReward
    }

    # Revert to optimal training that worked well before
    episodes_per_building = 12  # Previous successful setting

    model_kwargs = {
        'seed': 42,
        'batch_size': 256,
        'buffer_size': 50000,
        'learning_rate': 1e-4,    # Slightly higher than ultra-conservative
        'tau': 0.001,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'ent_coef': 0.005,        # Moderate exploration
        'target_entropy': -0.5,   # Balanced entropy target
        'learning_starts': 1000,
        'use_sde': False,
        'policy_kwargs': {
            'net_arch': [128, 128],
            'log_std_init': -3,   # Moderate action noise
        }
    }

    # Use the train_kwargs to specify any custom values that override defaults for parameters used to 
    # call the SAC learn  function(see https://stable-baselines3.readthedocs.io/en/master/modules/sac.html#stable_baselines3.sac.SAC.learn).
    learn_kwargs = {}

    SB3SAC.learn(
        schema=schema, 
        episodes_per_building=episodes_per_building, 
        env_kwargs=env_kwargs, 
        model_kwargs=model_kwargs,
        learn_kwargs=learn_kwargs
    )

if __name__ == "__main__":
    main()
