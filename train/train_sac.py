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


def create_sac_agent_from_saved_model(model_path: str, env) -> SAC:
    """
    Create a SAC agent and load a saved model
    
    Args:
        model_path: Path to the saved model (.zip file)
        env: CityLearn environment
        
    Returns:
        SAC agent with loaded model
    """
    print(f"[OK] Loading saved SAC model from: {model_path}")
    
    # Create agent
    agent = SAC(env)
    
    # Load the saved model
    agent.load(model_path)
    
    print(f"[OK] Successfully loaded SAC model")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="./data/schema.json", help="Path to CityLearn's schema.json")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_csv", type=str, default="./figures/sac_kpis_after_training.csv", help="Where to save KPIs CSV")
    parser.add_argument("--log_csv", type=str, default="./figures/sac_episode_rewards.csv", help="Where to save episode reward log")
    parser.add_argument("--plot_path", type=str, default="./figures/sac_reward_trend.png", help="Where to save reward trend plot")
    
    # ✨ NEW: Model saving/loading arguments
    parser.add_argument("--save-model", type=str, default="./models/sac_trained_model.zip", help="Path to save the trained model")
    parser.add_argument("--load-model", type=str, default=None, help="Path to load a pre-trained model (skips training)")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save models")
    
    args = parser.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()
    out_csv   = Path(args.log_csv)
    plot_path = Path(args.plot_path)
    kpi_csv   = Path(args.eval_csv)
    
    # ✨ NEW: Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)
    save_model_path = Path(args.save_model)

    # load base schema once
    with schema_path.open() as f:
        schema = json.load(f)
        
    schema["central_agent"] = True
    schema["episode_time_steps"] = 720            # short horizon for speed
    schema["rolling_episode_split"] = True        # roll through seasons
    schema["random_episode_split"] = False

    # 1- Environment init
    env = CityLearnEnv(schema=schema, random_seed=args.seed)
    
    # ✨ NEW: Check if we should load a pre-trained model or train from scratch
    if args.load_model:
        print(f"[OK] Loading pre-trained model from: {args.load_model}")
        agent = create_sac_agent_from_saved_model(args.load_model, env)
        print(f"[OK]  Skipping training, using loaded model")
    else:
        print(f"[OK] Training new SAC agent for {args.episodes} episodes")
        agent = SAC(env)
        agent.learn(episodes=args.episodes)
        
        # ✨ NEW: Save the trained model
        print(f"💾 Saving trained model to: {save_model_path}")
        agent.save(str(save_model_path))
        print(f"✅ Model saved successfully!")

    # 4- Episode logging on SEPARATE env (doesn't contaminate KPI evaluation)
    print(f"[OK] Running episode logging with {args.eval_episodes} episodes")
    start = schema.get("simulation_start_time_step", 0)
    end   = schema.get("simulation_end_time_step", 8759)
    log_schema = schema.copy()  # Copy to avoid mutating original
    log_schema["episode_time_steps"] = end - start + 1  # full span
    log_schema["rolling_episode_split"] = False
    log_schema["random_episode_split"] = False
    
    # Create separate logging env (doesn't affect KPI env)
    log_env = CityLearnEnv(schema=log_schema, random_seed=args.seed)
    log_agent = EnvActionAdapter(agent, log_env)
    log_df = run_episode_logging(log_env, log_agent, episodes=args.eval_episodes)
    
    # Save episode logs and plots
    log_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved episode logs to: {out_csv}")
    
    save_reward_plot(log_df, plot_path)
    print(f"[OK] Saved reward plot to: {plot_path}")
    
    # 5- EXACT OLD VERSION BEHAVIOR: Mutate schema but DON'T use it for KPI evaluation
    # The schema mutation in old version had NO EFFECT on save_kpis because env was already created
    schema["episode_time_steps"] = end - start + 1  # This mutation is IGNORED (like old version)
    schema["rolling_episode_split"] = False         # This mutation is IGNORED (like old version)
    schema["random_episode_split"] = False          # This mutation is IGNORED (like old version)
    
    # 6- Deterministic evaluation KPIs (EXACTLY like old version)
    eval_agent = EnvActionAdapter(agent, env)
    save_kpis(env, eval_agent, schema_path, kpi_csv)
    
    print(f"\n🎯 TRAINING SUMMARY:")
    print(f"   • Model saved to: {save_model_path}")
    print(f"   • Episode logs: {out_csv}")
    print(f"   • Reward plot: {plot_path}")
    print(f"   • KPIs saved to: {kpi_csv}")
    print(f"[OK] SAC training and evaluation finished!")


if __name__ == "__main__":
    main()