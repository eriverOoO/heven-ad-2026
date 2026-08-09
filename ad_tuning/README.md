# ad_tuning

Optuna contracts plus a guarded ROS 2/MORAI runner for Profile Stanley.
The live runner tunes the real `ad_planner` process; it does not contain a
second copy of the vehicle controller. SQLite remains the single-worker
fallback, while PostgreSQL coordinates a shared study across MORAI workers.

## Profile Stanley MORAI tuning

The launch file starts a tuning-only bringup with:

- MORAI command output enabled;
- vehicle description, GNSS/IMU localization, planner, and MORAI bridge;
- LiDAR/camera perception and RViz disabled to reduce simulator/worker load;
- DWA still configured as the production local obstacle planner, but
  `perception.enabled=false` during parameter comparison so intermittent
  OGM/DWA interventions do not contaminate the path-controller objective;
- `path_tracking.backend=profile_stanley`;
- a development bridge used only for MORAI `MultiEgoSetting` reset;
- the Optuna tuner.

Each trial runs this fail-closed sequence:

1. request `/ad/planner/hold_control` and wait until the vehicle stops;
2. atomically set the candidate ROS parameters;
3. reset Ego to the first global-path pose through MORAI;
4. reconstruct the path controller with `/ad/planner/reset_path_tracking`;
5. release the hold and measure completion, progress, CTE, speed, target speed,
   and brake command;
6. abort the trial on stale inputs, localization jumps, reverse heading,
   divergence, stall, or timeout.

Do not run the vehicle profiler, its loop guard, or another control publisher at
the same time. Start MORAI and load the intended map first, then run:

```bash
cd ~/heven_ad_2026_ws
export AD_DATA_DIR="$PWD/ad_data"
ros2 launch ad_tuning profile_stanley_morai_optuna.launch.py \
  morai_ip:=127.0.0.1 \
  maximum_trials:=30
```

Use `path_file:=path/...txt` to override the configured global path and
`course_length_m:=400.0` for a shorter repeatable tuning segment. A positive
`maximum_trials` is the shared study's completed-trial target, not the number
added by each invocation. Simulator or ROS infrastructure failures do not
consume this budget. Workers which were already evaluating can cause the final
count to exceed the target by at most roughly the active worker count. `0`
runs until interrupted.

Unless `output_dir` is set, results are written under:

```text
$AD_DATA_DIR/tuning/profile_stanley/
├── profile_stanley_optuna.sqlite3
└── workers/
    └── <worker-id>/
        ├── profile_stanley_trials.jsonl
        ├── profile_stanley_trajectories/
        ├── pending_database_results/
        ├── best_profile_stanley_optuna.yaml
        └── best_profile_stanley_optuna.json
```

Every worker exports a local snapshot of the global PostgreSQL winner. The
selected best YAML is left applied while that worker's planner remains held at
full brake. Release it only after reviewing the result. Validate the selected
winner once more with the normal perception-enabled DWA stack before promoting
it to the production planner configuration.

## Multi-computer PostgreSQL study

Each worker must run its own MORAI instance and ROS stack. PostgreSQL allocates
Optuna trials; it does not isolate simulator commands. Use a unique ROS domain
on a shared network and keep MORAI UDP endpoints local to each worker.

Install the PostgreSQL driver in the same Python environment used by ROS:

```bash
python3 -m pip install 'psycopg[binary]>=3.1,<4'
```

Create one PostgreSQL database and user on a stable host. Do not put the
password in a ROS YAML or launch command. Store it in a mode-600 `.pgpass`;
the worker environment can then use a URL without an inline password:

```bash
export OPTUNA_STORAGE_URL='postgresql+psycopg://ad_tuning@100.75.79.54:5432/heven_optuna'
export AD_TUNING_WORKER_ID='morai-pc-01'
export ROS_DOMAIN_ID=231

export AD_TUNING_SCENARIO='R_KR_PR_K-city_2025/2026_molit_comp_sample_scene'
export AD_TUNING_WEATHER='clear'
export AD_TUNING_MORAI_VERSION='S4.251001'
export AD_TUNING_CODE_REVISION="$(git -C src/heven_ad_2026 rev-parse HEAD)"
export AD_TUNING_VEHICLE_PROFILE_ID='20260727-ioniq5-accelerator40-brake20-v1'

ros2 launch ad_tuning profile_stanley_morai_optuna.launch.py \
  morai_ip:=127.0.0.1 \
  maximum_trials:=120
```

There is no designated warm-start PC. The first worker that reaches the
PostgreSQL study acquires an advisory lock and initializes its queue exactly
once; later workers immediately reuse it. If the newest compatible prior
study has at least 36 completed Profile Stanley trials (40 for DWA), its five
best feasible parameter sets are queued first and re-evaluated with the new
metric. The three static seeds follow. Old objective values and TPE history are
never mixed into the new study. The limits are configurable through
`warm_start.inherit_minimum_complete_trials` and
`warm_start.inherit_top_k`; setting either to zero disables inheritance.
Worker count and host choice therefore do not require a launch-argument
change.

The current three-computer deployment uses the Tailscale address of
`heven-right` (`100.75.79.54`) as the PostgreSQL server. The database is
`heven_optuna`, the role is `ad_tuning`, and the password is kept only in
each worker's mode-600 `.pgpass`. All workers source
`~/.config/heven/optuna-worker.env`; the shared non-secret URL is:

```bash
export OPTUNA_STORAGE_URL='postgresql+psycopg://ad_tuning@100.75.79.54:5432/heven_optuna'
```

The three workers use the same experiment values and storage URL, but distinct
worker IDs and ROS domains:

```bash
# heven-right
export AD_TUNING_WORKER_ID='heven-right'
export ROS_DOMAIN_ID=21

# heven-left
export AD_TUNING_WORKER_ID='heven-left'
export ROS_DOMAIN_ID=22

# heven-laptop
export AD_TUNING_WORKER_ID='heven-laptop'
export ROS_DOMAIN_ID=23
```

The study identity hashes all comparison-critical conditions:

- route and evaluated course length;
- search-space and objective versions;
- fixed controller parameters;
- scenario, weather, MORAI version, and vehicle profile;
- declared code revision and the installed `ad_tuning` source digest.

Changing any of them creates a different study automatically. Values marked
`unspecified` are allowed only for local SQLite use; a PostgreSQL worker
refuses to start until scenario, weather, and code revision are explicit. Use
the same real values on every distributed worker.

Distributed TPE uses worker-specific random seeds and `constant_liar=True`, so
workers avoid sampling near parameters which are already running. The live
runner uses `study.optimize()`, enabling Optuna RDB heartbeat. A worker which
dies leaves a stale trial that is marked failed after the configured grace
period and re-enqueued once by default.

Per-trial metrics, parameters, worker ID, experiment fingerprint, failure
reason, and trajectory location are stored in PostgreSQL trial attributes.
Full trajectory CSVs remain worker-local to avoid filling the database.
Before returning an objective value, each worker stages a JSON copy under
`pending_database_results/`. It is removed only after Optuna confirms the RDB
trial state; an unacknowledged file therefore remains available for audit when
the database or network fails.

The default storage behavior is:

- `OPTUNA_STORAGE_URL` set: shared PostgreSQL;
- variable absent: local SQLite under `output_dir`;
- database connection lost: full-brake hold, local result retention, and
  bounded reconnect attempts;
- process killed: RDB heartbeat failure plus one automatic retry.

## Search and objective

Optuna varies only controller-response parameters:

- cross-track, speed-softening, and heading gains;
- speed-dependent steering lookahead time;
- independent curvature lookahead;
- throttle PID `Kp/Kd`;
- brake PID `Kp/Kd`.

The 58.5 km/h target, 60 km/h ceiling, 6 m/s² lateral-acceleration limit,
lookahead bounds, physical acceleration/deceleration caps, and measured
longitudinal envelope remain fixed.

Completed trials minimize elapsed time, competition overspeed, speed above the
controller target, front- and rear-axle distance-weighted CTE MSE, unnecessary
braking, and sustained brake saturation. The deployed Profile Stanley tuning
configuration uses front and rear CTE weights of 300 each, in addition to the
hard 0.7 m boundary. Incomplete trials receive a large penalty while still
preserving progress information. A completed trial is feasible only when both
the front-axle and rear-axle maximum CTE are at most 0.7 m and no
reset/connectivity/abort fault occurred.

## Measured longitudinal envelope

`planner.yaml` contains the 2026-07-27 IONIQ 5 seed exported from:

- `accelerator-map-v1/profile.csv`, 40% accelerator;
- `brake-map-v2/profile.csv`, 20% brake.

The current 5–55 km/h table is interpolated by speed and extended at its edge.
Its measured braking delay is 0.11589156 s. Scalar acceleration/deceleration
parameters remain upper comfort/safety caps.

To regenerate the values after a new profile:

```bash
python3 scripts/export_longitudinal_profile.py \
  --accelerator-profile /path/to/accelerator/profile.csv \
  --accelerator-percent 40 \
  --brake-profile /path/to/brake/profile.csv \
  --brake-percent 20
```

The pure `run_study()` API and injectable `TrialRunner` remain available for
offline tests or another guarded simulator runner.

## DWA MORAI tuning

The DWA worker is intentionally separate from the Profile Stanley study. It
starts LiDAR perception and the real C++ DWA backend, requires a fresh
post-reset occupancy grid, and uses its own study/database:

```bash
cd ~/heven_ad_2026_ws
export AD_DATA_DIR="$PWD/ad_data"
ros2 launch ad_tuning dwa_morai_optuna.launch.py \
  morai_ip:=127.0.0.1 \
  maximum_trials:=120
```

Do not start this launch until MORAI has a repeatable obstacle scenario. A DWA
trial is feasible only if it completes without a collision, activates the local
planner for at least one second, and stays within the configured 4.5 m
primary-path envelope. A collision is an absolute constraint. Steering
smoothness is logged by the controller and used as an internal DWA critic, but
it is not added as a separate competition objective.

Optuna varies these ten response/critic parameters:

- goal, heading, clearance, command-continuity, primary-path, and speed weights;
- throttle PID `Kp/Kd`;
- brake PID `Kp/Kd`.

Vehicle geometry, the 100 m OGM, dynamic-window acceleration/deceleration,
steering-rate and lateral-acceleration limits, collision threshold, rollout
horizon, and sampling resolution remain fixed. This prevents the optimizer
from appearing successful by weakening collision safety or reducing sensor
coverage. The default DWA study uses 40 startup trials, multivariate TPE,
`constant_liar=True`, constraint-aware sampling, 120 completed trials, and a
150-trial total circuit breaker.

The static safety OGM consumes the raw LiDAR cloud and applies its own
`base_link` height/ego filters. Patchwork++ nonground output remains available
to clustering and tracking, but it is not allowed to erase a sparse far-range
box before the safety grid sees it. Inflation offsets and costs are
precomputed, so extending the grid to 100 m does not repeat `hypot`/`exp` for
every occupied cell on every scan.

The launch and objective have static/unit/build coverage only at this stage.
Far-range point returns, OGM timing, actual stopping distance, recovery, and
the final search ranges still require one controlled MORAI pilot before a
multi-computer search is considered validated.
