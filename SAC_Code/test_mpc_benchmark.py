#!/usr/bin/env python3
"""
Test the BenchmarkMPC (mpc_benchmark-main based) agent in CityLearn and print challenge KPIs.
"""

import os
import time
import numpy as np

from agents.mpc_agent import BenchmarkMPC
from rewards.custom_reward import ComfortandConsumptionReductionReward
from local_evaluation import create_citylearn_env, patch_building_for_evaluation, LSTMDynamicsAugmentor


def test_mpc_benchmark():
    print("Testing BenchmarkMPC (wrapped true MPC) baseline...")

    schema_path = os.path.join('data_saudi', 'schema.json')

    class Config:
        SCHEMA = schema_path

    env, wrapper_env = create_citylearn_env(Config, ComfortandConsumptionReductionReward, central_agent=True)
    dynamics = LSTMDynamicsAugmentor(schema_path)

    agent = BenchmarkMPC(wrapper_env, horizon=24, control_horizon=8)

    start_time = time.time()
    reset_result = env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        observations = dynamics.apply(observations, wrapper_env.observation_names)
        actions = agent.predict(observations)
        step_result = env.step(actions)

        if len(step_result) == 4:
            observations, reward, done, _ = step_result
        else:
            observations, reward, terminated, truncated, _ = step_result
            done = terminated or truncated

        total_reward += float(np.sum(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward)
        step_count += 1

    elapsed = time.time() - start_time

    env = patch_building_for_evaluation(env)
    metrics_df = env.evaluate_citylearn_challenge()

    print("\n=== BenchmarkMPC Results ===")
    print(f"Total reward over episode:    {total_reward:.4f}")
    print(f"Average reward per timestep:   {total_reward/step_count:.4f}")
    print(f"Agent wall-clock time:         {elapsed:.2f}s")
    print("\nChallenge metrics:")
    print(metrics_df)

    return metrics_df


if __name__ == "__main__":
    test_mpc_benchmark() 