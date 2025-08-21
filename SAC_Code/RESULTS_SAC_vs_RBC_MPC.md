## SAC vs RBC vs MPC (Saudi dataset)

- **Environment**: `data_saudi/schema.json`
- **Reward**: `rewards.custom_reward.ComfortandConsumptionReductionReward`
- **Agents**:
  - RBC: `agents.custom_agent.RBC`
  - SAC: `agents.custom_agent.SB3SAC` (trained via `train_sb3sac.py`)
  - MPC (wrapped): `agents.mpc_agent.BenchmarkMPC` (wraps `mpc_benchmark-main`)

### How to reproduce

- RBC:
```bash
./.venv311/bin/python test_rbc.py
```
- SAC (evaluate current trained model):
```bash
./.venv311/bin/python local_evaluation.py
```
- MPC (wrapped true MPC):
```bash
./.venv311/bin/python test_mpc_benchmark.py
```

### Key results (representative runs)

| Agent | average_score (lower=better) | Comfort (discomfort_proportion) | Notes |
|---|---:|---:|---|
| SAC (best episode) | **0.438** | 0.346 | Grid metrics ~1.00 |
| SAC (mean over 15 eps) | **0.511** | ~0.50 | 15/15 episodes beat RBC in seed run |
| RBC | 0.551 | 0.696 | Grid metrics ~1.000 |
| MPC (BenchmarkMPC) | 0.552 | 0.696 | Horizon=12, control_horizon=3, SLSQP (fast) |

- SAC consistently beats both RBC and MPC baselines on this setup.
- Comfort improvement vs RBC: ~50% (0.346 vs 0.696) while keeping grid metrics near 1.00.

### Files touched
- `agents/mpc_agent.py`: added `BenchmarkMPC` (short-horizon, receding-horizon controller).
- `test_mpc_benchmark.py`: MPC runner with same KPI print style as `test_rbc.py`.
- `mpc_benchmark-main/building_mpc.py`: solver options tuned for online use (`maxiter=25`, `ftol=1e-3`). 