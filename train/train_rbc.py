"""
Train RBC agent.

Usage:
  python train_rbc.py --schema ./data/schema.json 
"""

# ---- BEGIN hotfix: alias repo as 'corerl' ----
import importlib.util, sys, pathlib
import argparse
from pathlib import Path
import pandas as pd

project_root = pathlib.Path(__file__).parent.parent.resolve()
real_dir = project_root

# Alias top-level package corerl
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
from agents import RBC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="schema.json", help="Path to CityLearn's schema.json")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="kpis_after_training.csv", help="Where to save KPIs CSV")
    args = parser.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()
    episodes = args.episodes

    env = CityLearnEnv(schema=schema_path, central_agent=True, random_seed=args.seed)

    agent = RBC(env)

    agent.learn(episodes=episodes)

    # Deterministic evaluation + print KPIs
    kpis = env.evaluate()
    print(f"[OK] Training finished.")
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", None,
    ):
        print(kpis)
    
if __name__ == "__main__":
    main()
