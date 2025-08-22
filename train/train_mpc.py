"""
Train MPC agent.

Usage:
  python -m corerl.train.train_mpc --schema ./data/schema.json
  # or:
  python corerl/train/train_mpc.py --schema ./data/schema.json
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
from citylearn.citylearn import CityLearnEnv 
import time

from agents import MPC
from eval import run_episode_logging, save_reward_plot, save_kpis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="./data/schema.json", help="Path to CityLearn's schema.json")
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollout episodes to log")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="./figures/mpc_kpis_after_training.csv", help="Where to save KPIs CSV")
    parser.add_argument("--log_csv", type=str, default="./figures/mpc_episode_rewards.csv", help="Where to save episode reward log")
    parser.add_argument("--plot_path", type=str, default="./figures/mpc_reward_trend.png", help="Where to save reward trend plot")
    args = parser.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()
    out_csv   = Path(args.log_csv)
    plot_path = Path(args.plot_path)
    kpi_csv   = Path(args.eval_csv)

    # 1- Environment init
    env = CityLearnEnv(schema=schema_path, random_seed=args.seed)

    # 2- Agent init
    agent = MPC(env)

    # 3- Run episode logging (optional, similar to RBC)
    print(f"[INFO] Running {args.episodes} episodes for logging...")
    log_df = run_episode_logging(env, agent, episodes=args.episodes)
    
    # 4- Save episode logs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved episode logs to: {out_csv}")
    
    # 5- Save reward plot
    save_reward_plot(log_df, plot_path)
    print(f"[OK] Saved reward plot to: {plot_path}")

    # 6- Deterministic evaluation KPIs
    print(f"[INFO] Running KPI evaluation...")
    save_kpis(env, agent, schema_path, kpi_csv)
    
    # 7- Show MPC performance statistics
    mpc_stats = agent.get_performance_stats()
    print(f"\n[MPC PERFORMANCE]")
    print(f"MPC Success Rate: {mpc_stats['mpc_success_rate']:.1f}%")
    print(f"Total Optimization Calls: {mpc_stats['total_calls']}")
    print(f"Successful Optimizations: {mpc_stats['successful_optimizations']}")
    
    print(f"[OK] MPC training and evaluation finished.")


if __name__ == "__main__":
    main() 