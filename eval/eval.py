# corerl/eval/eval.py
from pathlib import Path
from typing import Any, Mapping, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import Agent

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
def run_episode_logging(env: CityLearnEnv, agent: Agent, episodes: int = 1) -> pd.DataFrame:
    """Run `episodes` episodes, return a DataFrame with reward stats per episode."""
    logs = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        step_rewards = []

        while not done:
            action = agent.predict(obs, deterministic=True)
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

def save_kpis(env: CityLearnEnv, agent, schema, kpi_csv: str, deterministic=True):
    """Fresh eval episode -> CityLearn KPIs -> CSV. Works for any agent by fixing action shape before env.step()."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    def _normalize_actions_for_env(actions, env: CityLearnEnv):
        """Return actions in the exact shape CityLearn expects."""
        # compute total dims & per-building dims
        per_b_dims = []
        total = 0
        for b in env.buildings:
            space = b.action_space
            if hasattr(space, "shape") and space.shape:
                d = int(np.prod(space.shape))
            elif hasattr(space, "n"):
                d = int(space.n)
            else:
                d = 1
            per_b_dims.append(d); total += d

        ca = getattr(env, "central_agent", False)

        # If central_agent=True CityLearn expects [[a0, a1, ..., a(total-1)]]
        if ca:
            # actions may be flat, or per-building [[...],[...],...]
            if isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], (list, tuple)):
                # concat per-building -> flat
                flat = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in actions], axis=0)
            else:
                flat = np.asarray(actions, dtype=np.float32).ravel()
            assert flat.size == total, f"central_agent expects {total} dims, got {flat.size}"
            return [flat.tolist()]

        # If central_agent=False CityLearn expects [[b0...], [b1...], ...]
        if isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], (list, tuple)):
            return [list(map(float, a)) for a in actions]

        # otherwise: split a flat vector per building
        flat = np.asarray(actions, dtype=np.float32).ravel()
        assert flat.size == total, f"multi-agent expects {total} dims, got {flat.size}"
        out, i = [], 0
        for d in per_b_dims:
            out.append(flat[i:i+d].tolist()); i += d
        return out

    print("[INFO] Starting KPI evaluation...")
    print("[INFO] Running evaluation episode...")
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False

    # optional sanity
    # print("[DEBUG] central_agent:", getattr(env, "central_agent", None))

    while not done:
        # Agent can return flat or per-building; we will normalize.
        actions = agent.predict(obs, deterministic=deterministic)
        env_actions = _normalize_actions_for_env(actions, env)  # <-- key fix

        obs, _, terminated, truncated, _ = env.step(env_actions)
        done = bool(terminated or truncated)

    print("[INFO] Evaluation episode completed. Attempting KPI calculation...")
    kpis = env.evaluate_citylearn_challenge()

    kpi_df = pd.DataFrame(kpis).T.reset_index()
    kpi_df.columns = ["kpi", "display_name", "weight", "value"]

    out = Path(kpi_csv); out.parent.mkdir(parents=True, exist_ok=True)
    kpi_df.to_csv(out, index=False)
    print(kpi_df)
    print(f"[OK] Saved KPIs to: {out.resolve()}")
    return kpi_df
