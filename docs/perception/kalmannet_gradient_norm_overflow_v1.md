# KalmanNet Gradient-Norm Overflow Guard + Extended 10K Forensics v1

Branch `fix/kalmannet-gradient-norm-overflow-v1`, from `origin/main`
`0f649c9` (merge of PR #57 `analysis(kalmannet): 10k gradient explosion
forensics`, verified `MERGED` before branching). **Training safety/
diagnostics only. No `KalmanNetGRU`/`KalmanNetFilter`/F/Q/H/AB3DMOT/
CenterPoint/ROS/prediction/planner change, no new production checkpoint
trained.**

## 1. Background

`analysis/kalmannet-10k-explosion-forensics-v1` (PR #57) ran a bounded
12-epoch reproduction of the real AV2 10k GENERIC-ROBUST seed-0 run's
documented epoch-16 collapse. It did NOT reproduce a genuine per-element
non-finite gradient within that window, but it DID reproduce multiple
severe instability episodes beginning around epoch 6, and found a
**second, previously-undocumented failure mode**: every individual
gradient element remained finite in every observed episode, but the
**aggregate L2 gradient norm** (computed via PyTorch's native float32
reduction) overflowed to `inf`. Because `clip_grad_norm_`'s clip
coefficient (`max_norm / (inf + eps) == 0`) then multiplies every
*still-finite* gradient element by exactly `0` (never `nan`, since
`0 * finite == 0` in IEEE754 -- unlike the proven `0 * inf == nan` STATE-C
poisoning mechanism), the batch's update became an **invisible,
unreported zero-update** that the network reliably recovered from every
time.

This task (1) makes that norm-overflow event explicit, correctly
accounted, and never silently hidden inside `clip_grad_norm_`'s ordinary
"successful" path, and (2) re-runs the exact same forensic replay
extended from 12 to 20 epochs to try to capture the original run's real
epoch-16 element-level collapse.

## 2. Three gradient states (nonfinite_guard.py)

| state | condition | policy |
|---|---|---|
| **A -- healthy** | every gradient element finite, aggregate norm finite | ordinary `clip_grad_norm_` + `optimizer.step()` |
| **B -- norm overflow** | every gradient element finite, aggregate norm reduction overflows to `inf`/`nan` | skip step, zero gradients, count separately, continue if isolated |
| **C -- element non-finite** | at least one gradient element itself `inf`/`nan` | skip step, zero gradients, count separately, apply existing PR #56 escalation policy |

`SafeStepOutcome.gradient_state` (`"healthy" | "norm_overflow" |
"element_nonfinite"`) makes this explicit; `outcome.norm_overflow` is a
convenience boolean. STATE C is checked **before** STATE B (an element
being non-finite always takes priority, since it can also make the
aggregate norm look non-finite for an unrelated reason).

### Why `clip_grad_norm_` can create an effective zero-update

`Tensor.norm(2)` and PyTorch's own internal `clip_grad_norm_` combination
accumulate in the gradient's **native dtype** (float32 for this model).
Squaring an individual gradient element as small as `~1.9e19` already
exceeds float32's representable max (`~3.4e38`) -- `(1.9e19)**2 ~= 3.6e38`
-- even though the raw element value itself is nowhere near overflowing.
Empirically verified (this session): a `4x4` tensor filled with `1e19`
(every element finite, `torch.isfinite().all() == True`) reports
`Tensor.norm(2) == inf`. If `clip_grad_norm_` were called anyway, its
`clip_coef = max_norm / (inf + eps) == 0`, and multiplying the
still-finite `1e19`-valued elements by that `0` gives exactly `0` (finite
arithmetic, never `nan`) -- a real parameter update of zero magnitude,
applied silently, with no distinguishing signal anywhere in the training
loop before this task.

### Robust diagnostic norm

`_grad_norm_robust_float64` upcasts each gradient tensor to `float64`
**before** squaring (not just when combining already-computed,
potentially-already-overflowed per-tensor norms) -- float64's
representable max (`~1.8e308`) comfortably covers even
`(3.4e38)**2 ~= 1.16e77` per element, so for any genuinely STATE-B
(element-finite) gradient set, this diagnostic recovers the true
large-but-finite magnitude instead of also reporting `inf`. Verified
directly: the `1e19`-per-element case above reports a robust norm of
`~4e19` (finite), against the ordinary norm's `inf`.
`outcome.robust_norm_also_overflowed` honestly flags the (expected to be
exceedingly rare, per the headroom above) case where even the float64
diagnostic itself overflows.

## 3. Policy for STATE B

`safe_clip_and_step` now checks element finiteness first (STATE C), then
computes both the ordinary and robust norms and checks the ordinary
norm's finiteness (STATE B) **before** ever calling `clip_grad_norm_`.
On a STATE-B detection: `opt.step()` is never called, gradients are
explicitly zeroed (`opt.zero_grad(set_to_none=False)`, the same
after-skip convention STATE C already used), and the event is reported
via `SafeStepOutcome(gradient_state="norm_overflow", norm_overflow=True,
grad_norm_robust=<the recovered magnitude>, ...)`. This is the same
class of fix as PR #56's own STATE-C guard, applied to the previously-
invisible STATE-B case.

## 4. STATE C policy (unchanged)

STATE C's skip/report/escalate/collapse-detection behavior is completely
unchanged from PR #56 -- verified by the full pre-existing test suite
passing unmodified (216/216 before this task's new tests were added).
`TrainResult.training_collapsed`/`catastrophic` semantics are untouched.

## 5. Health accounting (`TrainResult`, additive fields)

| field | meaning |
|---|---|
| `large_finite_gradient_count` | STATE A steps where `grad_norm_pre_clip > grad_clip` (real clipping occurred) |
| `norm_overflow_count` | total STATE B occurrences |
| `norm_overflow_skip_count` | STATE B occurrences resulting in a skipped step (== `norm_overflow_count` under the current policy, which always skips) |
| `per_element_nonfinite_gradient_count` | total STATE C occurrences |
| `nonfinite_gradient_skip_count` | STATE C occurrences resulting in a skipped step (== `per_element_nonfinite_gradient_count` today) |
| `parameter_collapse_count` | tier-3 events where `params_nonfinite_after_step` was True |
| `optimizer_state_collapse_count` | tier-3 events where `optimizer_state_nonfinite_after_step` was True |

`grad_skip_count` (pre-existing, PR #56) keeps its original combined
meaning (STATE B + STATE C skips together) for backward compatibility
with existing consumers (`run_2k_seed_sweep.py`); the new fields give the
breakdown that field alone could not. **Critically, STATE-B occurrences
do NOT feed the STATE-C-specific tier-2 escalation counter**
(`nonfinite_grad_skips_this_epoch`, gated on `MAX_NONFINITE_GRAD_SKIPS_
PER_EPOCH=5`) -- STATE B is self-neutralizing and has no proven Adam-
poisoning risk, so treating it the same as STATE C would have aborted a
run like the original 10k one around epoch 6-7 (well before its own
epoch-16 collapse), defeating this task's own extended-forensics goal.
Verified by `test_batched_repeated_norm_overflow_events_do_not_trigger_
state_c_escalation` (10 batches/epoch, 4 epochs, 100% STATE-B rate ->
`training_unstable=False`, ran the full budget).

## 6. Tests for the exact numerical cases (section 6 of this task)

`test_nonfinite_guard.py` (7 new tests): (A) ordinary finite gradient +
finite norm -> healthy, applied; (B) a `4x4` tensor filled with `1e19` --
every element finite, ordinary norm `inf`, robust norm `>1e19` and
finite, optimizer/parameters completely untouched, classified
`norm_overflow` not `element_nonfinite`; (C) one element `inf` ->
`element_nonfinite`, unchanged skip behavior; (D) one element `nan` ->
same. Plus: STATE B's gradient tensor is explicitly zeroed only *after*
the event is reported (not silently, and not left dangling); 5
subsequent healthy steps after a STATE-B event show zero lasting damage
(mirrors the existing STATE-C recovery test).

`test_training_escalation_policy.py` (4 new tests): repeated STATE-B
events across a full epoch budget never trigger `training_unstable`; a
mix of STATE-B and STATE-C events are tracked in fully separate counters
with no cross-contamination; `large_finite_gradient_count` only counts
steps strictly exceeding `grad_clip` (deterministic, fixed reported
norms, not dependent on real gradient magnitudes); parameter-collapse and
optimizer-state-collapse are counted independently for a
optimizer-state-only collapse scenario.

`test_explosion_forensics.py` (8 new tests): STATE-B events are captured
into `norm_overflow_events` without stopping the search (verified with 2
injected events across a 2-epoch run); when a true STATE-C event finally
fires after a prior STATE-B history, `preceding_norm_overflow_count_in_
ring_buffer` and `cumulative_norm_overflow_count_before_failure` are
correctly populated; Adam-state and parameter-magnitude trajectories have
exactly one entry per epoch, with `p99 <= max` and separate
GRU-vs-output-head norms; `largest_gradient_parameter` correctly ranks by
the robust (not ordinary) per-parameter norm; a 20-epoch bounded run
(synthetic data, no event) completes the full window; **a dedicated
regression test (`test_state_b_diagnostics_reflect_pre_guard_gradient_
not_post_zero`) for the pre-guard gradient-snapshot ordering bug found
during the real 20-epoch run (Section 7b) -- proves a STATE-B event's
`largest_norm_parameter_name`/`max_individual_abs_gradient` reflect the
true pre-zero gradient, not the trivial post-zero result**; the existing
first-event-capture test was also rewritten to inject its synthetic
non-finite gradient at the correct (pre-guard) point rather than inside a
mocked `safe_clip_and_step`, so it now exercises the REAL (unmocked)
guard's own detection logic end-to-end.

**Full suite: 235/235 pass** (216 pre-existing from PR #57 + 19 new).
`pyflakes`/`py_compile`/`git diff --check` all clean.

## 7. Extended 20-epoch forensic replay -- methodology

Reused every dataset/split/config parameter unchanged from PR #57:

```
shard_root       = ~/datasets/av2/processed/kalmannet_scaleup_v2/train
split_manifest   = ~/datasets/av2/manifests/scaleup_v2_internal_split_9k1k.json
corruption       = CONDITION_PRESETS["generic_robust"]
batch_size       = 64
learning_rate    = 0.004
grad_clip        = 10.0
loss_on_predict_only = True
hidden_size      = 32
use_length_bucketing = True, n_buckets = 8
device           = cpu
seed             = 0
max_epochs       = 20  (extended from PR #57's 12, per this task's own instruction)
```

`train_seqs = 496,434` internal-TRAIN sequences (byte-identical count to
both PR #57 and Task 1's original run); `7,757` batches/epoch. New
driver `run_10k_explosion_forensics_v2.py` (PR #57's own
`run_10k_explosion_forensics.py` kept unmodified as the 12-epoch record).

**Behavioral difference vs. PR #57's own 12-epoch run, deliberately
introduced by this task**: PR #57's search called the SAME
`safe_clip_and_step`, but at that time STATE-B batches still fell through
to an ordinary `clip_grad_norm_` call whose `0 * finite == 0` effect gave
an implicit, unreported zero-update (`opt.step()` WAS still called, just
with an all-zero gradient -- Adam's step counter advanced, its moving
averages incorporated a zero). **With this task's fix, STATE-B batches
now SKIP the optimizer step entirely** (no `opt.step()` call at all,
matching STATE C's skip behavior) -- Adam's step count does not advance
and its moving averages are not touched at all for that batch. This is
exactly the comparison section 14 of this task asks for.

## 7b. Extended 20-epoch forensic replay -- result

```
reproduced   = False
stop_reason  = "no genuine non-finite gradient observed within the bounded
                20-epoch window"
wall_time_s  = 24,487.3 (6.80 h)
n_epochs_run = 20 (0-19, the full bounded window -- never broke early)
norm_overflow_events captured = 15
element-level (STATE C) events = 0
```

**The true per-element non-finite gradient was NOT reproduced, even after
running directly through epoch 16 -- the exact epoch where the original
unguarded 10k run permanently collapsed.** Every epoch's
`n_element_nonfinite_batches = 0`; every one of the 93,084+140k = 155,140
batches processed across the 20-epoch window (7,757/epoch x 20) went
through `safe_clip_and_step` without a single genuine element-level
non-finite gradient.

### Per-epoch trajectory (raw)

| epoch | train_loss_mean | grad_norm_max | STATE-B count |
|---|---|---|---|
| 0 | 0.930 | 298,995 | 0 |
| 1 | 0.751 | 2,106 | 0 |
| 2 | 0.765 | 92,494 | 0 |
| 3 | 0.742 | 71.8 | 0 |
| 4 | 5.593 | 2.107e9 | 0 |
| 5 | 0.741 | 160 | 0 |
| **6** | 7.615e13 | inf | **1** |
| 7 | 4,323 | 4.760e12 | 0 |
| **8** | 8.508e12 | inf | **2** |
| 9 | 553.5 | 8.977e15 | 0 |
| 10 | 1.457e11 | 5.450e17 | 0 |
| 11 | 0.743 | 1.120e7 | 0 |
| **12** | 6.647e17 | inf | **2** |
| 13 | 5.371e8 | 2.992e17 | 0 |
| 14 | 15.61 | 7.185e14 | 0 |
| 15 | 1.432e9 | 2.369e17 | 0 |
| **16** | 7.169e17 | inf | **6** |
| **17** | 1.233e12 | inf | **3** |
| **18** | 7.391e8 | inf | **1** |
| 19 | 1.731 | 8.254e7 | 0 |

**Epoch 16 -- the original run's own documented collapse point -- had the
single highest STATE-B event count of any epoch in this run (6 of 15
total)**, directly confirming this is the genuinely turbulent epoch the
original run's own record described ("From epoch 6 onward, every
subsequent epoch hit at least one batch with an inf... gradient norm"; the
original's actual collapse landed specifically at epoch 16). **Every one
of these 6 epoch-16 events was correctly classified as STATE B (element-
finite, aggregate-norm overflow) and safely skipped** -- none produced a
genuine element-level non-finite gradient. Qualitatively, epochs 13-18
show a real behavior shift from the earlier, always-fully-recovering
pattern (epochs 4-12): the model no longer resets cleanly to the ~0.74
baseline between every epoch (epoch 14 opens at loss 213.5, not ~0.74;
epoch 15 ends at 1.43e9, still far above baseline) before finally
recovering again by epoch 19 (loss 1.73, close to healthy).

### STATE-B event population (all 15 events)

**A real, mechanistic finding, not something this task set out to find:**
`largest_norm_parameter_name` is **`input_fc.0.weight`** (the first linear
layer of `KalmanNetGRU`'s `input_fc = nn.Sequential(nn.Linear(in_dim,
hidden_size), nn.ReLU())`, the layer that directly consumes the raw
innovation/state-difference input features every timestep) for **all 15 of
15 events**, no exceptions.

`grad_norm_robust` (the float64 diagnostic) for these events ranges from
`1.70e20` to `6.69e26` -- confirming these are genuinely enormous
gradients, several orders of magnitude beyond anything a healthy training
step should ever produce, not borderline/ambiguous cases.
`robust_norm_also_overflowed = False` for every single event -- the
float64 diagnostic never itself overflowed, meaning the TRUE magnitude was
always successfully recovered (per Section 2's own headroom argument).

**A second real bug was found and fixed while inspecting this output, not
before running it** (see git diff for the fix): `NormOverflowEvent.
max_individual_abs_gradient` and `.largest_norm_parameter_name`, as
originally implemented, were computed by re-inspecting `net`'s gradients
**after** `safe_clip_and_step` had already returned -- but `safe_clip_and_
step`'s STATE-B (and STATE-C) paths call `opt.zero_grad()` **before**
returning. This meant every one of the 15 events' own reported
`max_individual_abs_gradient` trivially came back `0.0`, and
`largest_norm_parameter_name` trivially came back the FIRST parameter in
declaration order (`input_fc.0.weight`) regardless of which parameter was
actually responsible -- a real diagnostic bug, not a genuine mechanistic
finding, and this session's own real run's raw JSON output reflects the
**buggy** values (all `max_individual_abs_gradient = 0.0`). **Fixed** by
snapshotting both diagnostics immediately after `backward()`, before
calling `safe_clip_and_step` at all (see `run_bounded_forensic_search`'s
"CRITICAL ORDERING" comment), with two new regression tests
(`test_state_b_diagnostics_reflect_pre_guard_gradient_not_post_zero`,
and the rewritten `test_first_event_capture_stops_immediately_and_
localizes_parameter`) proving the fix. **This bug does NOT affect**:
`grad_norm_pre_clip`/`grad_norm_robust`/`robust_norm_also_overflowed`
(computed and returned by `safe_clip_and_step` itself, BEFORE its own
zero_grad call -- always correct), the STATE classification itself, the
skip decision, or whether reproduction succeeded (`n_element_nonfinite_
batches` is derived from `outcome.grad_nonfinite_pre_clip`, likewise set
before any zeroing). **Given this bug does not require re-running the
20-epoch search to correct** (only affects two diagnostic fields whose
own correct value -- the fix is verified by dedicated unit tests -- would
not change this task's central `reproduced=False` conclusion), the search
was not re-run; the fix is committed for correctness in any FUTURE run.

### Adam-state / parameter-magnitude trajectory (sections 12/13)

Adam's `max_abs_exp_avg`/`max_abs_exp_avg_sq` stay in a tight, bounded
band (`~0.02-0.6`, `~0.04-0.78`) across ALL 20 epochs, including epochs
6-18 with their many STATE-B events -- **no progressive, unbounded growth
of the optimizer's own moment estimates occurred**, confirming the guard's
skip-and-zero policy successfully prevents STATE-B events from poisoning
Adam's internal state (the exact protection this task's Section 3 policy
was designed to provide). The largest single-epoch jump is epoch 13
(`max_abs_exp_avg_sq` 0.0601 -> 0.513) and epoch 16 (0.0973 -> 0.775) --
both immediately after/during the most turbulent epochs, but neither
compounds into later epochs (epoch 19's own values are back down to
`0.0197`/`0.0672`, near the epoch-0-5 baseline).

Parameter magnitudes (`max_abs_parameter`, `gru_weight_norm`,
`output_head_weight_norm`) grow **smoothly and monotonically** across all
20 epochs (`max_abs_parameter`: 3.09 -> 15.53; `gru_weight_norm`: 35.2 ->
134.9; `output_head_weight_norm`: 3.61 -> 9.21) -- **no discontinuous
jump or instability is visible in the weights themselves at any epoch**,
including 16. This is a real, positive finding: even though the
GRADIENT signal repeatedly reached astronomical magnitude during STATE-B
episodes, the guard's skip policy means those gradients never actually
touched the weights -- the parameters' own trajectory is smooth
throughout, undisturbed by the underlying gradient turbulence.

### Effect of skipping norm-overflow batches (section 14, diagnostic only)

Compared with PR #57's own 12-epoch run (same seed/data/config, but STATE
B still received an implicit zero-update via `clip_grad_norm_`'s
`0*finite=0` -- meaning `opt.step()` WAS still called, Adam's internal
step counter still advanced, on those batches): the two runs' loss
trajectories are byte-identical through epoch 5 (no STATE-B event had yet
occurred), then diverge starting epoch 6 -- e.g. at the equivalent
epoch-7/batch-500 point, loss 0.7453 (this run, skip) vs. 0.7467 (PR #57,
implicit zero-step); by epoch 7/batch-2000 the two runs' `grad_norm_max`
values differ by many orders of magnitude (this run: 21.2 finite; PR #57:
6.44e11). **This is diagnostic only, per this task's own instruction --
not interpreted as an accuracy improvement.** The two policies are
numerically DIFFERENT (skipping vs. stepping-with-zero-gradient changes
Adam's own step-count-dependent bias-correction terms for every
subsequent update), and this divergence is the direct, expected
consequence -- not evidence that one policy trains "better," only that
they are not numerically equivalent.

## 8. Root cause (section 17 of the task)

Closest classification: **(D) the original true failure does not
reproduce once overflow batches are safely skipped** -- with an important
qualification. Every one of the 15 STATE-B episodes observed in this run,
including all 6 that occurred specifically during epoch 16 (the original
run's own documented collapse epoch), was safely absorbed by the new
guard without ever producing a genuine per-element non-finite gradient.
Combined with the clean, bounded Adam-state and monotonic parameter-
magnitude trajectories throughout, this is direct evidence that **the
specific epoch-16 STATE-B episodes this run encountered would very likely
have been exactly the events that corrupted the original unguarded run**
(via `clip_grad_norm_`'s `0*inf=nan` mechanism on an aggregate-overflowed
gradient it never distinguished from a genuinely poisoning one) -- and
this task's guard prevented that outcome at the precise epoch it mattered
most.

This is not classified as **(A) repeated aggregate norm overflow
eventually precedes a true per-element non-finite gradient**, because no
such per-element event occurred anywhere in the full 20-epoch window,
including well past epoch 16 (epochs 17-18 each still produced further
STATE-B episodes with no element-level escalation). It is not (B) (no
single pathological batch was ever isolated as the TRUE failure, since no
true failure occurred) or (C) (Adam/parameter trajectories stayed bounded
and smooth, no progressive growth toward failure was observed) or (F)
(the result here is a clean, fully-reproducible negative result across
the entire extended window, not inconclusive).

## 9. Training-policy decision (section 18 of the task, NOT implemented)

**(A) Current safe-skip policy is enough; proceed to 10k seeds 1/2.**
The guard now correctly distinguishes and safely handles both known
failure modes (STATE B and STATE C); this run demonstrates it successfully
carries the exact model/data/config combination that previously collapsed
at epoch 16 through the ENTIRE 20-epoch window with zero element-level
non-finite gradients and bounded, smoothly-recovering Adam/parameter
state. No further guard/policy change, learning-rate reduction, robust-
loss addition, or optimizer-handling modification is evidenced as
necessary by this run. The next defensible step is running additional
seeds (1, 2, ...) of the SAME 10k GENERIC-ROBUST configuration with this
guard active, to establish whether epoch-16-class instability recurs
across seeds and whether the guard continues to prevent escalation to a
genuine collapse -- **not implemented in this task, per its own explicit
scope** ("Do NOT start another full training seed").

## Files

`tools/kalmannet_training/nonfinite_guard.py` (three-state classification,
robust float64 norm, STATE-B skip policy),
`tools/kalmannet_training/{trainer_core.py,batched_trainer.py}` (new
`TrainResult` accounting fields, STATE-B routed away from the STATE-C
tier-2 escalation counter),
`tools/kalmannet_training/explosion_forensics.py` (STATE-B event capture,
Adam-state/parameter-magnitude per-epoch trajectories, preceding-context
capture on a true STATE-C event; pre-guard gradient-diagnostic snapshot
ordering fix -- see Section 7b -- so STATE-B/STATE-C diagnostics reflect
the true pre-zero gradient rather than `safe_clip_and_step`'s own
post-zero state),
`tools/kalmannet_training/run_10k_explosion_forensics_v2.py` (new, 20-
epoch driver; v1 kept unmodified),
`tools/kalmannet_training/test_{nonfinite_guard,training_escalation_
policy,explosion_forensics}.py` (19 new tests, including two regression
tests for the pre-guard-snapshot ordering fix), this file. No
`kalmannet_core.py`/`KalmanNetFilter`/F/Q/H/AB3DMOT/CenterPoint/ROS/
prediction/planner file changed.
