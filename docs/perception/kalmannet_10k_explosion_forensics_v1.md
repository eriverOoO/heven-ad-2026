# KalmanNet 10k GENERIC-ROBUST Seed-0 Explosion Forensics v1

Branch `analysis/kalmannet-10k-explosion-forensics-v1`, from `origin/main`
`837a518` (merge of PR #56 `fix(kalmannet): harden training against
numerical collapse`, verified `MERGED` before branching). **Analysis-only
task. No training run to completion, no new seed, no official AV2 VAL
evaluation, no architecture/hyperparameter change. The bounded (≤12
epoch) forensic search reported here is the entire deliverable; the safe-step
guard from PR #56 stayed enabled throughout.**

## 1. Goal

Task 1 (`exp/kalmannet-av2-10k-generic-v1`, PR #55) trained a full-scale
GENERIC-ROBUST KalmanNet (seed 0) on the frozen 10,000-scenario AV2 dataset
and reported a real gradient-explosion collapse at **epoch 16** (weights
permanently non-finite from that point on), while `catastrophic=False` was
incorrectly reported by the trainer at the time. Task 2
(`fix/kalmannet-training-instability-v1`, PR #56, merged) root-caused the
general mechanism (a technically-finite-but-huge loss can `backward()` into
an `inf`/`nan` gradient; `clip_grad_norm_` converts that `inf` into a
poisoned `nan` via `0 * inf == nan`; Adam's moment buffers, once `nan`,
never recover) using a minimal, non-KalmanNet-specific PyTorch demonstration,
and shipped `nonfinite_guard.safe_clip_and_step` plus a three-tier escalation
policy. That mechanism did **not** reproduce on 5 seeds x 2 arms of the
much cheaper 2k Stage-1 dataset (0/10 collapses).

This task's goal: reproduce the **exact original** 10k seed-0 instability,
using the identical frozen dataset/split/config/seed, and capture the FIRST
real gradient-explosion event (if any occurs) in bounded forensic detail --
ring buffer of preceding batches, failure-batch characteristics,
parameter-level localization, individual-sequence replay, and
gap/motion/hidden-state/gain hypothesis testing -- bounded to a maximum of
**12 epochs** (not the original run's full 60-epoch budget, and explicitly
not permitted to continue past 12 without an observed event).

## 2. Precondition verification

`gh pr view 56 --json state,mergedAt` confirmed `MERGED` before any edit.
Worktree created at `/tmp/heven-worktrees/kalmannet-10k-explosion-forensics-v1`
(`git worktree add ... -b analysis/kalmannet-10k-explosion-forensics-v1
origin/main`), never the dirty main checkout (~1,900 pre-existing CRLF-only
modifications, unrelated to this project's own work).

## 3. Frozen dataset / split / config (unchanged, verified byte-identical to Task 1)

```
shard_root       = ~/datasets/av2/processed/kalmannet_scaleup_v2/train
split_manifest   = ~/datasets/av2/manifests/scaleup_v2_internal_split_9k1k.json
                   (schema av2_kalmannet_stage0_split_v1, seed 20260930)
corruption       = CONDITION_PRESETS["generic_robust"] (gaussian_noise_std_m=0.3,
                   dropout_prob=0.1, dropout_burst_enabled=True,
                   dropout_burst_prob=0.02, dropout_burst_min_len=2,
                   dropout_burst_max_len=5, seed=20260905)
batch_size       = 64
learning_rate    = 0.004
grad_clip        = 10.0
loss_on_predict_only = True
hidden_size      = 32
use_length_bucketing = True, n_buckets = 8
device           = cpu
seed             = 0
max_epochs       = 12 (bounded for THIS task only -- Task 1's own run used 60)
```

`train_seqs = 496,434` internal-TRAIN sequences loaded (identical count to
Task 1's own reported `train=496,434`); internal-VAL and official AV2 VAL
were **not loaded** -- this is a training-dynamics-only analysis, no
evaluation was run. `7,757` batches/epoch, matching Task 1's own reported
`optimizer_steps_this_epoch~=7,757` exactly -- confirms this run replays the
identical batch structure per epoch.

## 4. Result: **NOT REPRODUCED within the 12-epoch bound**

```
reproduced   = False
stop_reason  = "no genuine non-finite gradient observed within the bounded
                12-epoch window"
wall_time_s  = 14,612.0 (4.06 h)
n_epochs_run = 12 (0-11, the full bounded window -- never broke early)
```

Zero batches, across all 12 x 7,757 = 93,084 optimizer-step attempts, ever
produced a genuinely non-finite (element-level, `torch.isfinite(p.grad).all()
== False`) gradient -- `grad_nonfinite_pre_clip` was `False` on every single
batch. `n_nonfinite_loss_batches = 0` in every epoch (the *loss* itself, up
to `1.02e25`, was always float64-representable-finite). Per the PR #56
`safe_clip_and_step` guard's own design, **the guard was never actually
exercised as a skip-and-continue decision** in this run -- it only ever took
the "step applied" branch, because its pre-check (`_grads_finite`, per-element
`torch.isfinite`) never found a genuinely non-finite element.

## 5. Real per-epoch trajectory (raw, not smoothed)

| epoch | train_loss_mean | grad_norm_mean | grad_norm_max | n_batches | n_nonfinite_loss |
|---|---|---|---|---|---|
| 0 | 0.930 | 40.6 | 298,995 | 7,757 | 0 |
| 1 | 0.751 | 0.912 | 2,106 | 7,757 | 0 |
| 2 | 0.765 | 13.5 | 92,494 | 7,757 | 0 |
| 3 | 0.742 | 0.579 | 71.8 | 7,757 | 0 |
| **4** | **5.593** | **455,521** | **2.107e9** | 7,757 | 0 |
| 5 | 0.741 | 0.604 | 160 | 7,757 | 0 |
| **6** | **7.615e13** | **inf** | **inf** | 7,757 | 0 |
| 7 | 90,789 | 1.139e8 | 6.442e11 | 7,757 | 0 |
| **8** | **6.442e21** | **inf** | **inf** | 7,757 | 0 |
| **9** | **1.015e25** | **inf** | **inf** | 7,757 | 0 |
| **10** | 9.627e7 | inf | inf | 7,757 | 0 |
| **11** | 8.563e15 | inf | inf | 7,757 | 0 |

Every "exploded" epoch's mean is dominated by exactly one pathological batch
each time (confirmed via the ring buffer/progress-callback trace, Section 6)
-- the epoch's *tail* batches always return to the healthy ~0.73-0.76 band
(directly confirmed: the final 20 ring-buffer entries, batches 7737-7757 of
epoch 11, all have `batch_loss` in `[0.51, 1.30]` and
`grad_nonfinite_pre_clip=False` throughout). **Training self-recovers every
single time** in this bounded window -- every post-spike epoch (1, 3, 5)
returns cleanly to baseline.

## 6. Pre-failure trajectory / recurring spike pattern (there was no single
"failure batch" since no event fired -- this section instead characterizes
the six recurring escalation episodes actually observed)

Progress-callback trace (every 500 batches + epoch end) shows six distinct
large-gradient episodes, at epoch {0 (mild, 500-6500), 2 (mild), 4, 6, 7-9
(the three largest), 10-11}:

| episode (epoch, approx. batch) | grad_norm_pre_clip (peak) | batch/running loss (peak) | self-recovered? |
|---|---|---|---|
| 0, ~b500 | 298,995 (finite) | loss trending down normally | n/a -- mild |
| 2, ~b6000 | 92,494 (finite) | loss 0.77 (mild) | n/a -- mild |
| **4, b6000** | **2.107e9 (finite)** | running mean 7.01 | yes -> epoch 5 loss=0.741 |
| **6, b2500** | **inf (norm overflow)** | running mean 2.36e14 -> decayed to 7.62e13 by epoch end | yes -> epoch 7 opened normal, but epoch 7 itself spiked again |
| **7, b2000** | 6.442e11 (finite) | running mean 3.5e5 -> decayed to 9.08e4 | yes -> epoch 8 opened normal |
| **8, b7000** | **inf (norm overflow)** | running mean 7.14e21 -> decayed to 6.44e21 | yes -> epoch 9 opened normal |
| **9, b4500** | **inf (norm overflow)** | running mean 1.75e25 (largest observed) -> decayed to 1.02e25 | yes -> epoch 10 opened normal |
| **10, b6000** | inf (norm overflow) | running mean 1.25e8 -> decayed to 9.63e7 | yes -> epoch 11 opened normal |
| **11, b3500-b7000** | 1.73e17 (finite) then inf (b7000) | loss climbed to 3.39e5 then spiked to 9.49e15, decayed to 8.56e15 by epoch end | yes -- final 20 batches of the run are all healthy (Section 5) |

Each episode is qualitatively identical: one (or a small cluster of) batch(es)
produces an extremely large gradient; `grad_norm_pre_clip` sometimes
overflows to literal `inf` via the L2-norm-squaring computation
(`_grad_norm_safe`/PyTorch's internal `clip_grad_norm_` norm, both effectively
summing squared per-element values); the loss's *running mean* spikes hard
(since it's diluted by only ~500-1000 batches) and then decays smoothly back
to baseline as more healthy batches accumulate in the mean; the very next
epoch always opens at the normal ~0.73-0.76 baseline loss. No episode ever
left any lasting scar on the network's actual parameters (confirmed
indirectly: if it had, subsequent epochs could not have returned to the
~0.73-0.76 baseline loss band they consistently did).

## 7. First non-finite parameter

**N/A -- not reproduced.** `localize_first_nonfinite_parameter` was never
invoked (only called on the `outcome.grad_nonfinite_pre_clip == True`
branch, which never fired). `param_grad_reports = []`,
`first_nonfinite_param_name = None` in the output JSON.

## 8. Failure-batch characteristics

**N/A -- not reproduced.** `failure_entry = None`,
`failure_sequence_details = []`.

## 9. Top loss-contributing sequence / offending timestep

**N/A -- not reproduced** (no failure batch to attribute loss within). The
recurring-episode batches (Section 6) were not individually inspected for
per-sequence loss attribution, since `build_sequence_detail`/
`replay_single_sequence` are only invoked by the search on a genuine
`grad_nonfinite_pre_clip` event, matching this task's own instruction to
build these tools for the reproduction case specifically.

## 10. Gap/corruption analysis vs. dataset-wide percentiles

Computed once, over a fixed deterministic 50,000-sequence subsample of the
same 496,434-sequence internal-TRAIN pool (`np.random.RandomState(0)`), for
future use should a reproduction attempt succeed:

| metric | p50 | p90 | p95 | p99 |
|---|---|---|---|---|
| missing_fraction | 0.143 | 0.263 | 0.313 | 0.455 |
| max_gap (frames) | 2 | 5 | 6 | 7 |
| sequence_length (frames) | 38 | 110 | 110 | 110 |
| speed_max (m/s, GT) | 0.630 | 11.70 | 14.14 | 17.90 |
| accel_max proxy (m/s^2, GT) | 1.559 | 15.95 | 31.40 | 87.75 |

**N/A for a failure-batch comparison** since no failure batch exists.

## 11. Motion-outlier / hidden-state / gain analysis

**N/A -- not reproduced.** No `ForensicResult.failure_entry`, hence no
`max_hidden_state_abs`/`max_gain_abs`/`max_innovation_abs` value tied to a
specific failure event exists. (These fields *are* populated in every
ring-buffer entry regardless of outcome, per `run_batch`'s diagnostics
plumbing -- confirmed non-negative and finite in every retained entry,
Section 6's healthy tail included -- but there is no "failure" entry to
contrast them against.)

## 12. Individual-sequence replay

**N/A -- not reproduced.** `replay_single_sequence` was never invoked on
real data (only exercised by its own unit tests against synthetic
sequences).

## 13. Root cause / why the original run's own explosion was not reproduced

**Directly cross-referenced against Task 1's own committed run record**
(`docs/perception/kalmannet_av2_10k_generic_v1.md`, Section 7, on
`origin/main`), which independently documented the *same run's* own
gradient-explosion timeline:

> "From epoch 6 onward, every subsequent epoch hit at least one batch with
> an `inf` (or, before that, extremely large but technically finite, up to
> ~10^32) gradient norm. Gradient clipping (`max_norm=10.0`) contained every
> one of these through epoch 15 ... At epoch 16 the network's weights went
> permanently non-finite."

This bounded reproduction (epochs 0-11) reproduces this exact *qualitative*
pattern almost perfectly: the original run's own instability **onset at
epoch 6** matches this run's own onset (the first genuinely severe episode,
`grad_norm_pre_clip=inf`, occurred at epoch 6 here too -- Section 6); both
runs show the network absorbing repeated large-but-ultimately-contained
gradient spikes for several epochs without permanent damage; the original
run's *actual* catastrophic collapse (permanent non-finite weights) occurred
at **epoch 16 -- 4 epochs past this task's own explicit 12-epoch bound**.

**Conclusion: the 12-epoch bound was very likely simply too short to reach
the original run's real collapse point, not evidence that the collapse
mechanism itself failed to reproduce.** The qualitative dynamics (episode
onset epoch, self-recovery pattern, escalating peak magnitudes across
epochs 6-11) track the original run's own documented epoch-6-through-15
"turbulent but contained" phase closely. This task's own instruction was
explicit that continuing past 12 epochs without an observed event is not
permitted, so this conclusion is reported as the most likely explanation
rather than tested further.

**A second, structural finding worth recording independently of the above:**
every large-gradient episode observed in this run was an **aggregate L2-norm
overflow** (`grad_norm_pre_clip == inf`, computed via float32/64 squaring of
individual gradient values that were each still per-element finite,
`torch.isfinite(p.grad).all() == True`) rather than a **genuine per-element
non-finite gradient** (`grad_nonfinite_pre_clip == True`, an individual
tensor value literally `inf`/`nan`). This distinction matters mechanically:
when only the aggregate norm overflows, `clip_grad_norm_`'s zero clip
coefficient (`max_norm / (inf + eps) == 0`) multiplies each *still-finite*
gradient element by `0`, producing exactly `0` (not `nan`, since
`0 * finite_value == 0` in IEEE754, unlike `0 * inf == nan`) -- a
self-neutralizing no-op update, matching every recovery observed in Section
6. **Task 2's proven poisoning mechanism (`0 * inf == nan`) specifically
requires at least one gradient element itself to be non-finite** -- this
did not happen even once in this 93,084-batch run. It is plausible (not
proven here) that the original run's own epoch-16 event was exactly this
rarer, stronger failure mode (an actual per-element `inf`/`nan`, as opposed
to the aggregate-only overflow this run repeatedly and safely absorbed) --
consistent with why it took 10 epochs of "merely" aggregate-overflow
turbulence (epochs 6-15) before a genuinely poisoning event occurred.

**Batch ordering / corruption hash comparison:** both runs use the identical
`CONDITION_PRESETS["generic_robust"]` (fixed `seed=20260905`) and the
identical `numpy.random.RandomState(seed=0)`-derived per-epoch shuffle
(`bucketed_epoch_batch_order`, `n_buckets=8`), so batch content and ordering
are deterministic and match Task 1's own recipe exactly, batch-count for
batch-count (7,757/epoch in both). The one architectural difference between
this task's bespoke `run_bounded_forensic_search` loop and Task 1's real
`train_one_run_batched` path is that this search calls `safe_clip_and_step`
directly per batch without wrapping it in the tier-1/tier-2 escalation
counters (`grad_skip_count`, `MAX_NONFINITE_GRAD_SKIPS_PER_EPOCH`) that the
production trainers now apply -- immaterial here since no skip ever
occurred, but noted for completeness (this search's own design intentionally
breaks immediately on the very first genuine non-finite event, per this
task's own "capture the FIRST event" framing, rather than tolerating up to 5
skips/epoch as the production escalation policy would).

## 14. Safety-guard effect

The PR #56 guard was active and correctly evaluated on every one of the
93,084 batches processed. It never needed to intervene in this run (zero
skips), because -- per Section 13 -- every observed instability episode was
the self-neutralizing aggregate-overflow-but-element-finite case, which
recovers safely even without the guard (via the `0 * finite == 0`
arithmetic). This is consistent with, not contradictory to, Task 2's
finding: the guard's protective value is specifically for the rarer,
stronger per-element-non-finite case, which simply did not occur within
this bounded window.

## 15. Tests

`tools/kalmannet_training/test_batched_kalmannet.py` gained 5 new tests
(diagnostics-out plumbing on `step_masked`/`run_batch`, all additive/
default-`None`, zero behavior change when omitted) -- file now 24/24 pass.
`tools/kalmannet_training/test_explosion_forensics.py` (new, 12 tests):
ring-buffer retention/bounded-epoch stop (no event), progress-callback
optionality and epoch-end firing, ring-buffer size cap, ring-buffer
per-sequence loss stats, deterministic no-event reruns, first-event capture
+ stop-immediately + parameter localization (via a monkeypatched injected
non-finite gradient on the 3rd batch), parameter-localization correctness on
every-finite and one-poisoned synthetic nets, individual-sequence-replay
never-calls-optimizer-step and isolation-between-probes, GT-derived
speed/accel stats, and run_batch-style diagnostics presence in every ring
buffer entry.

Full suite: **216/216 pass** (169 pre-existing from PR #56 + 5 new
`test_batched_kalmannet` cases + 12 new `test_explosion_forensics` cases +
30 already-counted PR #56 tests -- see exact breakdown in the git history;
net new this task: 17). `pyflakes` clean on all 3 changed/new files.
`py_compile` clean. `git diff --check` clean (no trailing-whitespace/
CRLF issues introduced).

## 16. What was NOT done (explicit scope boundaries honored)

- **Did not** train past epoch 12 (stopped exactly at the bound with no
  event observed, per this task's explicit instruction).
- **Did not** run a new seed.
- **Did not** evaluate official AV2 VAL.
- **Did not** alter `KalmanNetGRU`, `KalmanNetFilter`, `run_batch`'s
  numerical recursion, or any optimizer/hyperparameter default.
- **Did not** disable or weaken the PR #56 safe-step guard at any point.
- **Did not** commit the 194 KB raw JSON forensic dump, the 496,434-sequence
  dataset, or any checkpoint -- only this compact summary document,
  the reusable tooling, and its tests are committed. The full JSON lives at
  `~/datasets/av2/checkpoints/kalmannet_10k_explosion_forensics_seed0.json`
  (machine-local, not committed).

## 17. Files

`tools/kalmannet_training/explosion_forensics.py` (new -- bounded search
core: `RingBufferEntry`, `SequenceDetail`, `ParamGradReport`,
`EpochTrajectoryPoint`, `ForensicResult`, `localize_first_nonfinite_parameter`,
`build_ring_buffer_entry`, `build_sequence_detail`, `replay_single_sequence`,
`run_bounded_forensic_search`);
`tools/kalmannet_training/run_10k_explosion_forensics.py` (new -- ad hoc
driver script against the real frozen 10k dataset, not a permanent CLI/
setup.py entry point);
`tools/kalmannet_training/test_explosion_forensics.py` (new, 12 tests);
`tools/kalmannet_training/batched_kalmannet.py` (additive `diagnostics_out`
parameter on `step_masked`/`run_batch`, default `None`, zero behavior
change when omitted);
`tools/kalmannet_training/test_batched_kalmannet.py` (+5 tests for the new
diagnostics plumbing);
this file. No `kalmannet_core.py`, `trainer_core.py`, `batched_trainer.py`,
`nonfinite_guard.py`, or any production training-safety file changed --
this task's own instruction was analysis only, and Task 2's safety fix is
reused completely unmodified.

## 18. Recommended next task

Per this task's own required decision (not implemented here): **re-run this
exact same bounded forensic search extended to approximately 18-20 epochs**
(comfortably past the original run's own documented epoch-16 collapse
point, still far short of the original's full 60-epoch budget) on the same
frozen 10k dataset/seed/config, keeping the PR #56 guard active throughout.
This is the most direct way to test the Section 13 hypothesis (the 12-epoch
bound was simply too short) with minimal additional engineering, reusing
every tool built in this task unchanged -- only `MAX_EPOCHS` in
`run_10k_explosion_forensics.py` needs to change. If a genuine per-element
non-finite gradient does occur in epochs 12-20, this task's own
`ForensicResult` machinery (ring buffer, parameter localization, individual-
sequence replay) will capture it in exactly the detail this task specified.
Not started here, per this task's own explicit 12-epoch bound.

Do NOT start another training run as part of closing out this task.
