"""
Train RBC agent.

Usage:
  python train_rbc.py --schema ./data/schema.json 
"""

from citylearn.citylearn import CityLearnEnv 
import argparse
from pathlib import Path
from citylearn.agents.rbc import RBC as Agent 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="schema.json", help="Path to CityLearn's schema.json")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="kpis_after_training.csv", help="Where to save KPIs CSV")
    args = parser.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()

    env = CityLearnEnv(schema=schema_path, central_agent=False, random_seed=args.seed)
    agent = Agent(
        env,
        batch_size=256
    )

    # Train
    agent.learn(episodes=args.episodes)

    # Deterministic evaluation + export KPIs
    kpis = env.evaluate()
    kpis.to_csv(args.eval_csv, index=True)
    print(f"[OK] Training finished. KPIs saved to: {args.eval_csv}")


if __name__ == "__main__":
    main()
