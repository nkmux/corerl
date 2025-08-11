"""
Train RBC agent.

Usage:
  python -m corerl.train.train_rbc --schema ./data/schema.json
  # or:
  python corerl/train/train_rbc.py --schema ./data/schema.json
"""

# ---- BEGIN hotfix: alias repo as 'corerl' ----
import importlib.util, sys, pathlib
import argparse
from pathlib import Path

project_root = pathlib.Path(__file__).parent.parent.resolve()
real_dir = project_root

spec = importlib.util.spec_from_file_location(
    "corerl",
    real_dir / "__init__.py",
    submodule_search_locations=[str(real_dir)]
)
corerl = importlib.util.module_from_spec(spec)
sys.modules["corerl"] = corerl
spec.loader.exec_module(corerl)
# ---- END hotfix ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the helpers that actually exist
from corerl.eval.eval import run_episode_logging, save_reward_plot, save_kpis

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import HourRBC as Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="./data/schema.json", help="Path to CityLearn's schema.json")
    parser.add_argument("--episodes", type=int, default=5, help="Number of rollout episodes to log")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="figures/kpis_after_training.csv", help="Where to save KPIs CSV")
    parser.add_argument("--log_csv", type=str, default="figures/episode_rewards.csv", help="Where to save episode reward log")
    parser.add_argument("--plot_path", type=str, default="figures/reward_trend.png", help="Where to save reward trend plot")
    args = parser.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()
    out_csv   = Path(args.log_csv)
    plot_path = Path(args.plot_path)
    kpi_csv   = Path(args.eval_csv)

    env = CityLearnEnv(schema=schema_path, central_agent=False, random_seed=args.seed)
    agent = Agent(env)

    # RBC typically doesn't need learning; try if it's available
    if hasattr(agent, "learn"):
        try:
            agent.learn(episodes=args.episodes)
        except TypeError:
            agent.learn()

    # 1) Run & log episodes
    log_df = run_episode_logging(env, agent, episodes=args.episodes)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved reward log to: {out_csv.resolve()}")

    # 2) Plot episode returns
    save_reward_plot(log_df, plot_path)

    # 3) Deterministic evaluation KPIs
    save_kpis(env, kpi_csv)

    print("[OK] Finished.")


if __name__ == "__main__":
    main()