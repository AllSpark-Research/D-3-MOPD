# D³-MOPD example

Minimal reference layout for launching D³-MOPD (Domain-adaptive Dynamic
Distillation with Multi-teacher On-Policy Distillation) on top of slime.

## Files

- `run.sh` — reference launch script; every path, model, and data field is a
  placeholder. Fill the `<...>` values in and hand the resulting env off to
  slime's async training entrypoint via your cluster orchestrator.

## Three-part architecture

```
                        wandb
                          │
                          │ poll rollout/opd_reverse_kl/{domain},
                          │      rollout/domain_score/{domain}
                          ▼
                ┌──────────────────┐
                │ tools/d3mopd/    │
                │     watcher.py   │─── writes ──▶ ${D3MOPD_STATUS_PATH}
                │  (external proc) │                     │
                └──────────────────┘                     │ reads per batch
                                                         ▼
     ┌─────────────┐    per-sample     ┌───────────────────────────────────┐
     │ N teachers  │ ◀── HTTP route ──│  training (slime)                  │
     │ (SGLang     │      by domain    │   ├── DynamicRatioStratified      │
     │  servers)   │                   │   │      RolloutDataSource         │
     └─────────────┘                   │   ├── multi_teacher_distillation  │
                                       │   │      reward_func              │
                                       │   └── d3mopd_rollout_log          │
                                       └───────────────────────────────────┘
```

- **Data source** (`slime_plugins.data_sources.stratified`) — reads the mixture
  ratios the watcher writes, produces strict per-batch per-domain quotas via
  largest-remainder rounding.
- **Reward path** (`slime.rollout.multi_teacher_distillation`) — routes each
  sample to `OPD_DOMAIN_TEACHER_ROUTES[sample.metadata[route_key]]`.
- **Logger** (`slime_plugins.logging.d3mopd_rollout_log`) — rolls per-
  `data_source` counts and scores up into per-domain wandb metrics that the
  watcher then polls.
- **Watcher** (`tools/d3mopd/watcher.py`) — external process, restart-safe.

## Watcher modes

| `--mode`             | signal                                        | when to use                                      |
|----------------------|-----------------------------------------------|--------------------------------------------------|
| `static`             | KL plateau + reward stagnation → binary DOWN  | filter-based rejection sampling                   |
| `static-ratio`       | same detector, emits `mixture_ratio` instead  | one-shot per-domain quota shrink                  |
| `dynamic`            | initial-KL-normalized gap: `ema/initial_kl`   | continuous, gap-only                              |
| `dynamic-delta`      | velocity: `max(0, -ΔEMA/EMA_earlier)`         | continuous, velocity-only                         |
| `dynamic-composite`  | `normalized_gap × progress-velocity`          | default recommendation                            |

Two robustness knobs on `dynamic-composite`:

- `--progress-floor` — retains a gap-only fallback when KL rebounds
  (velocity ≤ 0). Default 0 = paper-exact.
- `--rehearsal-domains` + `--abs-kl-floor` — pins rehearsal-style domains
  (frozen-student teacher, `initial_kl ≈ 0`) onto the ratio floor instead of
  letting their exploded normalized gap hog the mixture.

## Integration surface

Env vars read by the plugins + watcher:

| var                          | consumer                    | purpose                                    |
|------------------------------|-----------------------------|--------------------------------------------|
| `OPD_DOMAIN_TEACHER_ROUTES`  | reward path                 | `domain=url|...` teacher URL mapping       |
| `OPD_TEACHER_ROUTE_KEY`      | reward path                 | metadata key to route by (default `data_source`) |
| `D3MOPD_DOMAIN_MAP`          | data source + logger + watcher | `domain=ds1,ds2|...` domain→data_source map |
| `D3MOPD_STATUS_PATH`         | data source + filter + watcher | shared JSON path                        |
| `D3MOPD_DATA_SOURCE_PATH`    | slime CLI                   | which stratified data source class to wire |
| `D3MOPD_FILTER_PATH`         | slime CLI (static mode)     | which downsample filter to wire            |
| `D3MOPD_RATIO_JITTER`        | data source                 | per-batch multiplicative jitter α ∈ [0,1)  |
| `D3MOPD_REHEARSAL_DOMAINS`   | watcher                     | Patch B: comma-separated rehearsal domains |

CLI paths passed to slime:

- `--data-source-path $D3MOPD_DATA_SOURCE_PATH`
- `--custom-rm-path slime.rollout.multi_teacher_distillation.reward_func`
- `--custom-reward-post-process-path slime.rollout.multi_teacher_distillation.post_process_rewards`
- `--custom-log-rollout-function-path slime_plugins.logging.d3mopd_rollout_log.log_rollout_data`
- `--dynamic-sampling-filter-path $D3MOPD_FILTER_PATH` (only in static mode)
