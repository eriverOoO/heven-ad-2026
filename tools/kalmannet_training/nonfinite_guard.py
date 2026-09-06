"""Safe optimizer-step guard for the KalmanNet trainers.

**Root-caused bug this module fixes** (see
``docs/perception/kalmannet_training_instability_v1.md`` for the full
forensic writeup): neither ``trainer_core.train_one_run`` nor
``batched_trainer.train_one_run_batched`` checked gradient finiteness
before calling ``torch.nn.utils.clip_grad_norm_``/``optimizer.step()``.
Both only checked that the *loss* was finite before ``backward()`` --
but a loss can be finite (even merely large, e.g. ~1e13-1e32, as observed
in the real AV2 10k GENERIC-ROBUST run) while its *gradient*, computed
through ``backward()``'s chain rule across a long recurrent sequence,
overflows to ``inf``/``nan``. Once that happens:

1. ``torch.nn.utils.clip_grad_norm_`` does **not** neutralize an ``inf``
   gradient into a safe zero update -- verified directly (see
   ``test_nonfinite_guard.py::test_clip_grad_norm_on_inf_gradient_produces_nan``):
   the total norm of a gradient set containing an ``inf`` entry is itself
   ``inf``, so the library's own clip coefficient becomes
   ``max_norm / (inf + eps) == 0``, and multiplying an ``inf``-containing
   tensor by that zero coefficient computes ``0 * inf == nan`` per IEEE754
   -- clipping *converts* an ``inf`` gradient into a worse, silently
   "successfully clipped-looking" ``nan`` gradient.
2. ``optimizer.step()`` (Adam) then applies that ``nan`` gradient to its
   ``exp_avg``/``exp_avg_sq`` moment buffers. Once a moment buffer entry
   is ``nan``, it stays ``nan`` forever -- ``beta * nan + (1-beta) *
   anything == nan`` for every future step regardless of how many
   subsequent gradients are perfectly finite (verified by
   ``test_adam_state_is_permanently_poisoned_by_one_nonfinite_gradient``).
   The corresponding parameter entries become ``nan`` and never recover.

This module's ``safe_clip_and_step`` fixes this: it inspects gradients
for finiteness **before** calling ``clip_grad_norm_`` at all, and skips
the step entirely (zeroing gradients, never touching the optimizer)
whenever a non-finite gradient is found -- exactly the guard the
instability forensics found is missing. It also verifies, defensively,
that parameters and optimizer state remain finite after every step that
IS applied (this should be unreachable given the pre-check, but is
checked anyway so a genuinely new failure mode fails loudly rather than
silently).

**No estimator/architecture change.** This module only wraps the
optimizer-step call; it never touches ``KalmanNetGRU``, ``KalmanNetFilter``,
or any analytical model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

# Escalation policy threshold shared by both trainers (see
# docs/perception/kalmannet_training_instability_v1.md "Escalation
# Policy" for the full derivation). Conservative, not tuned against any
# specific run's outcome:
#
# - A handful of isolated non-finite-gradient events per epoch (e.g. one
#   pathological long-gap sequence in an otherwise healthy batch/sequence
#   mix) is skip-and-continue: the guard above already prevents them from
#   corrupting anything, so there is no safety reason to abort training
#   over a rare outlier.
# - More than this many non-finite-gradient skips within a SINGLE epoch
#   signals the run is systematically unstable, not merely unlucky once --
#   training is aborted after that epoch (not mid-epoch, so the epoch's
#   own EpochRecord is still recorded truthfully).
MAX_NONFINITE_GRAD_SKIPS_PER_EPOCH = 5


@dataclass(frozen=True)
class SafeStepOutcome:
    """Result of one guarded optimizer step attempt."""

    applied: bool                              # True iff optimizer.step() was actually called
    grad_norm_pre_clip: float                  # may be inf/nan -- reported for diagnostics regardless
    grad_nonfinite_pre_clip: bool
    grad_nonfinite_post_clip: bool
    params_nonfinite_after_step: bool = False
    optimizer_state_nonfinite_after_step: bool = False

    @property
    def skipped_reason(self) -> str | None:
        if self.applied:
            return None
        if self.grad_nonfinite_pre_clip:
            return "nonfinite_gradient_pre_clip"
        if self.grad_nonfinite_post_clip:
            return "nonfinite_gradient_post_clip"
        return "unknown"

    @property
    def collapsed(self) -> bool:
        """True iff this step left the model in a permanently non-finite
        state -- should be unreachable given the pre-checks above, but is
        the hard, unambiguous fail-fast signal callers must check."""
        return self.params_nonfinite_after_step or self.optimizer_state_nonfinite_after_step


def _grad_norm_safe(params: list[torch.nn.Parameter]) -> float:
    """Same L2-combination convention as ``trainer_core._grad_norm`` --
    reused-equivalent, not duplicated logic drift: sum of squared
    per-parameter norms, square-rooted. Deliberately does NOT special-case
    inf/nan -- callers want the true diagnostic value (inf/nan is itself
    meaningful signal), only the *decision* to step is gated elsewhere."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return math.sqrt(total)


def _all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    for t in tensors:
        if t is not None and not torch.isfinite(t).all():
            return False
    return True


def _grads_finite(params: list[torch.nn.Parameter]) -> bool:
    return _all_finite(p.grad for p in params if p.grad is not None)


def optimizer_state_finite(opt: torch.optim.Optimizer) -> bool:
    """Checks every tracked Adam moment buffer (``exp_avg``/``exp_avg_sq``)
    across every parameter group -- the exact state the forensics found
    gets permanently poisoned by one non-finite gradient."""
    for state in opt.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(key)
            if tensor is not None and not torch.isfinite(tensor).all():
                return False
    return True


def safe_clip_and_step(
    net: torch.nn.Module,
    opt: torch.optim.Optimizer,
    grad_clip: float | None,
) -> SafeStepOutcome:
    """Call AFTER ``loss.backward()``, INSTEAD OF calling
    ``torch.nn.utils.clip_grad_norm_``/``optimizer.step()`` directly.

    Guarantees: ``optimizer.step()`` is never called with a non-finite
    gradient. Parameters and optimizer state are checked finite after any
    step that IS applied; a caller observing
    ``outcome.collapsed == True`` must fail training immediately (see
    ``docs/perception/kalmannet_training_instability_v1.md`` "Escalation
    Policy") -- this module does not itself abort training, it only
    reports the state so the caller's escalation policy can decide.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    grad_norm_pre_clip = _grad_norm_safe(params)
    grad_nonfinite_pre_clip = not _grads_finite(params)

    if grad_nonfinite_pre_clip:
        opt.zero_grad(set_to_none=False)
        return SafeStepOutcome(
            applied=False, grad_norm_pre_clip=grad_norm_pre_clip,
            grad_nonfinite_pre_clip=True, grad_nonfinite_post_clip=False,
        )

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)

    grad_nonfinite_post_clip = not _grads_finite(params)
    if grad_nonfinite_post_clip:
        # Defensive: unreachable in practice (clipping a finite gradient
        # set only ever scales it down or leaves it unchanged), but
        # checked explicitly per this task's own instruction rather than
        # assumed.
        opt.zero_grad(set_to_none=False)
        return SafeStepOutcome(
            applied=False, grad_norm_pre_clip=grad_norm_pre_clip,
            grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=True,
        )

    opt.step()

    params_finite_after = _all_finite(params)
    opt_state_finite_after = optimizer_state_finite(opt)

    return SafeStepOutcome(
        applied=True, grad_norm_pre_clip=grad_norm_pre_clip,
        grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
        params_nonfinite_after_step=not params_finite_after,
        optimizer_state_nonfinite_after_step=not opt_state_finite_after,
    )
