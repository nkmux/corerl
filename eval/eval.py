# corerl/eval/eval.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- internals ----------
def _sum_reward(r):
    """Collapse CityLearn reward (scalar/dict/list/ndarray) into a single float."""
    if isinstance(r, dict):
        vals = []
        for v in r.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals.append(np.sum(v))
            else:
                vals.append(v)
        return float(np.sum(vals))
    if isinstance(r, (list, tuple, np.ndarray)):
        return float(np.sum(r))
    return float(r)

def _is_done(terminated, truncated):
    """Support bool or dict done flags."""
    def _any(x):
        if isinstance(x, dict): return any(x.values())
        return bool(x)
    return _any(terminated) or _any(truncated)

# ---------- public API ----------
def run_episode_logging(env, agent, episodes: int) -> pd.DataFrame:
    """Run `episodes` episodes, return a DataFrame with reward stats per episode."""
    logs = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        ep_return = 0.0
        step_rewards = []

        while not done:
            action = agent.predict(obs) if hasattr(agent, "predict") else agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            r = _sum_reward(reward)
            ep_return += r
            step_rewards.append(r)
            done = _is_done(terminated, truncated)

        step_rewards = np.asarray(step_rewards, dtype=float)
        logs.append({
            "episode": ep,
            "episode_return": ep_return,
            "steps": int(step_rewards.size),
            "reward_mean": float(step_rewards.mean()) if step_rewards.size else 0.0,
            "reward_var": float(step_rewards.var(ddof=0)) if step_rewards.size else 0.0,
            "reward_std": float(step_rewards.std(ddof=0)) if step_rewards.size else 0.0,
            "reward_min": float(step_rewards.min()) if step_rewards.size else 0.0,
            "reward_max": float(step_rewards.max()) if step_rewards.size else 0.0,
        })
        print(f"[Episode {ep}] return={ep_return:.4f} "
              f"mean={logs[-1]['reward_mean']:.6f} "
              f"std={logs[-1]['reward_std']:.6f} steps={logs[-1]['steps']}")
    return pd.DataFrame(logs)

def save_reward_plot(log_df: pd.DataFrame, out_path: Path) -> None:
    """Save a line plot of episode_return vs episode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["episode"], log_df["episode_return"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward (Return)")
    plt.title("Episode Reward Trend")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Plot saved to: {out_path.resolve()}")

def save_kpis(env, out_csv: Path) -> None:
    """Run deterministic env.evaluate() and save KPIs to CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    kpis = env.evaluate()
    print(kpis)
    pd.DataFrame(kpis).to_csv(out_csv, index=False)
    print(f"[OK] KPIs saved to: {out_csv.resolve()}")
