# Contributing

D³-MOPD is released as a research artifact accompanying the paper. Contributions
that improve correctness, reproducibility, or clarity are welcome; please open
an issue first for anything larger than a small bug fix so we can align on
scope before you invest time.

## Guidelines

- **Bug reports** — please include a minimal reproduction (or a failing test).
  Runtime issues in the D³-MOPD-specific code paths (`slime/rollout/multi_teacher_distillation.py`,
  `slime_plugins/data_sources/stratified.py`, `slime_plugins/filters/d3mopd_downsample_filter.py`,
  `slime_plugins/logging/d3mopd_rollout_log.py`, `tools/d3mopd/watcher.py`) are
  the most likely to get quick attention.
- **Bug fixes** — PRs should add or extend a unit test under `tests/test_d3mopd_*.py`
  demonstrating the fix.
- **New features / algorithms** — please discuss in an issue first.
- **Underlying framework issues** — D³-MOPD is built on
  [slime](https://github.com/THUDM/slime); issues that reproduce with vanilla
  slime (no D³-MOPD plugins wired) are best reported upstream.

## Running the tests

```bash
python -m pytest tests/test_d3mopd_dynamic_unit.py \
                 tests/test_d3mopd_dynamic_delta_unit.py \
                 tests/test_d3mopd_composite_unit.py -v
```

These are pure-function tests — no wandb, no GPUs, no cluster required.
