"""
Train SAC agent.

Usage:
  python -m corerl.train.train_sac 
  # or:
  python corerl/train/train_sac.py 
"""

# ---- BEGIN hotfix: alias repo as 'corerl' ----
import json
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

from agents import SAC
from eval import run_episode_logging, save_reward_plot, save_kpis

# ---- put these helpers in the same file (above main) ----
import numpy as np

def _to_env_actions(actions, env):
    """Normalize policy output to what CityLearn expects in env.step()."""
    # total dims helper
    def _total_dims():
        tot = 0
        for b in env.buildings:
            space = b.action_space
            if hasattr(space, "shape") and space.shape:
                tot += int(np.prod(space.shape))
            elif hasattr(space, "n"):
                tot += int(space.n)
            else:
                tot += 1
        return tot

    if getattr(env, "central_agent", False):
        # expect [[flat...]]
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            flat = arr.tolist()
        else:
            # list-of-lists -> concatenate
            flat = np.concatenate([np.asarray(a).ravel() for a in actions]).tolist()
        assert len(flat) == _total_dims(), f"central_agent expects {_total_dims()} dims, got {len(flat)}"
        return [flat]
    else:
        # expect [[b0...], [b1...], ...]
        if isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], (list, tuple)):
            return [list(a) for a in actions]
        # flat -> split per building
        arr = np.asarray(actions, dtype=np.float32).ravel()
        out, i = [], 0
        for b in env.buildings:
            space = b.action_space
            if hasattr(space, "shape") and space.shape:
                d = int(np.prod(space.shape))
            elif hasattr(space, "n"):
                d = int(space.n)
            else:
                d = 1
            out.append(arr[i:i+d].tolist()); i += d
        return out

class EnvActionAdapter:
    """Wrap any agent so .predict(obs, det) returns env-ready action shape."""
    def __init__(self, base_agent, env):
        self.base = base_agent
        self.env  = env
    def predict(self, obs, deterministic=True):
        a = self.base.predict(obs, deterministic=deterministic)
        return _to_env_actions(a, self.env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="./data/schema.json", help="Path to CityLearn's schema.json")
    parser.add_argument("--episodes", type=int, default=100, help="Number of rollout episodes to log")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="./figures/kpis_after_training.csv", help="Where to save KPIs CSV")
    parser.add_argument("--log_csv", type=str, default="./figures/episode_rewards.csv", help="Where to save episode reward log")
    parser.add_argument("--plot_path", type=str, default="./figures/reward_trend.png", help="Where to save reward trend plot")
    args = parser.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()
    out_csv   = Path(args.log_csv)
    plot_path = Path(args.plot_path)
    kpi_csv   = Path(args.eval_csv)
    training_episodes = args.episodes 

    # load base schema once
    with schema_path.open() as f:
        schema = json.load(f)
        
    schema["central_agent"] = True
    schema["episode_time_steps"] = 720            # short horizon for speed
    schema["rolling_episode_split"] = True        # roll through seasons
    schema["random_episode_split"] = False

    # 1- Environment init
    env = CityLearnEnv(schema=schema, random_seed=args.seed)
    agent = SAC(env)
    agent.learn(episodes=training_episodes)

    # 4- Run & log episodes
    start = schema.get("simulation_start_time_step", 0)
    end   = schema.get("simulation_end_time_step", 8759)
    schema["episode_time_steps"] = end - start + 1  # full span (e.g., 8760 or 8670)
    schema["rolling_episode_split"] = False
    schema["random_episode_split"] = False
    
    # 6- Deterministic evaluation KPIs
    eval_agent = EnvActionAdapter(agent, env)
    save_kpis(env, eval_agent, schema_path, kpi_csv)
    print(f"[OK] Finished.")



if __name__ == "__main__":
    main()