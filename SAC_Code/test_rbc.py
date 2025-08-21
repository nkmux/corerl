#!/usr/bin/env python3
"""
test_rbc.py

Test the RBC baseline performance for comparison against your learned agents.
"""

import os
import time
import numpy as np

from agents.custom_agent import RBC
from rewards.custom_reward import ComfortandConsumptionReductionReward
from local_evaluation import create_citylearn_env, patch_building_for_evaluation, LSTMDynamicsAugmentor


def test_rbc():
    """Run one episode of the RBC agent and report metrics."""
    print("Testing RBC baseline...")

    # 1) Point to upgraded original schema.json
    schema_path = os.path.join('data_saudi', 'schema.json') # put data_saudi to use saudi data

    # 2) Create a minimal config object for the helper
    class Config:
        SCHEMA = schema_path

    # 3) Instantiate full CityLearnEnv + metadata-only wrapper with central_agent=True
    env, wrapper_env = create_citylearn_env(Config, ComfortandConsumptionReductionReward, central_agent=True)
    dynamics = LSTMDynamicsAugmentor(schema_path)

    # 4) Instantiate your RBC (rule-based) agent with just the wrapper
    agent = RBC(wrapper_env)
    agent.comfort_band = 0.1  # tighten band so dynamics influence actions

    # 5) Run one episode
    start_time = time.time()
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        observations = reset_result[0]
    else:
        observations = reset_result
    
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        # apply dynamics to observations before prediction
        observations = dynamics.apply(observations, wrapper_env.observation_names)
        actions = agent.predict(observations)
        step_result = env.step(actions)
        
        # Handle different step return formats
        if len(step_result) == 4:
            observations, reward, done, _ = step_result
        else:
            observations, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
            
        # reward may be a list (per building) or a single float or numpy array
        if isinstance(reward, (list, np.ndarray)):
            reward_sum = float(np.sum(reward))
        else:
            reward_sum = float(reward)
            
        total_reward += reward_sum
        step_count += 1

    elapsed = time.time() - start_time

    # 6) Fetch the official challenge metrics DataFrame with patching
    env = patch_building_for_evaluation(env)
    metrics_df = env.evaluate_citylearn_challenge()

    # 7) Print results
    print("\n=== RBC Baseline Results ===")
    print(f"Total reward over episode:    {total_reward:.4f}")
    print(f"Average reward per timestep:   {total_reward/step_count:.4f}")
    print(f"Agent wall-clock time:         {elapsed:.2f}s")
    print("\nChallenge metrics:")
    print(metrics_df)

    return metrics_df


if __name__ == "__main__":
    test_rbc()
