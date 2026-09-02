# Archive

Optimizers that were explored but produced none of the params in
`../best_params_all_games.json`. Kept for provenance, not for reuse.

| File | Why it's here |
|---|---|
| `run_game.py` | Superseded by `../opt.py`, which is a fork of it with parallel per-coordinate evaluation. The overnight runs in `../logs/output_*.log` use `opt.py`'s `FINAL: best=... verified_mean=...` line, not this file's `FINAL BEST: ...`. |
| `optimize_all.py` | Multi-game driver with averaged evaluation (`evaluate_mean`). Never completed a run — no `results_optimized_*.json` exists. |
| `fast_coord.py` | Cached coordinate descent. Both runs died mid-search; the stubs are in `../logs/crashed/`. |
| `targeted_optimizer.py` | CantStop-specific search. Never completed a run. |
| `advanced_optimizer.py` | May-era. Produced `../results/results_advanced_*.json`, superseded by the July runs. |
| `evolutionary.py` | May-era GA. No surviving results. |
| `fast_optimizer.py` | May-era. No surviving results. |
| `rl_optimizer.py` | REINFORCE-based search (commit cb8a302). No surviving results. |
| `main.py` | Original entry point, superseded. |
| `param_space.py` | Dependency of `advanced_optimizer.py`, `evolutionary.py` and `main.py` only. |

These do a bare `import api_client`, which resolves from the repo root. Running
them from here needs the root on `sys.path`:

    PYTHONPATH=.. .venv/bin/python archive/<script>.py
