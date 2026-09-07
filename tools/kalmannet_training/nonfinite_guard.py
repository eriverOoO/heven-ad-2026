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

**Second failure mode, found by the 10k explosion forensics** (see
``docs/perception/kalmannet_gradient_norm_overflow_v1.md``): a batch can
have every individual gradient *element* finite, yet the **aggregate L2
norm reduction itself** overflows to ``inf``. This happens because
``Tensor.norm(2)`` (and PyTorch's own internal ``clip_grad_norm_``
combination) accumulates in the gradient's native dtype (float32 here) --
squaring an individual element as small as ``~1.9e19`` already exceeds
float32's representable max (``~3.4e38``), even though the raw element
value itself is nowhere near overflowing. This module distinguishes this
**STATE B (norm overflow, elements finite)** from the more dangerous
**STATE C (an element itself is non-finite)** explicitly:

- STATE A (healthy): every element finite, aggregate norm finite --
  ordinary clip + step.
- STATE B (norm overflow): every element finite, but the ordinary
  (float32) aggregate norm reduction overflowed to ``inf``/``nan``. Calling
  ``clip_grad_norm_`` here would silently compute
  ``clip_coef = max_norm / (inf + eps) == 0`` and multiply every
  *still-finite* gradient element by that zero -- ``0 * finite == 0``
  exactly (never ``nan``, unlike the STATE C case), so parameters/Adam
  state are never at risk here, but the resulting zero-update would be
  completely invisible to any caller not specifically looking for it. This
  guard skips the step explicitly instead and reports the event, rather
  than letting it hide inside an ordinary-looking "successful" step.
- STATE C (element non-finite): the known, more dangerous Adam-poisoning
  hazard (see the mechanism above) -- skip and report, unchanged from the
  original guard's behavior.

A **robust, float64-accumulated diagnostic norm**
(``_grad_norm_robust_float64``) is always additionally computed whenever
every element is finite (STATE A or STATE B), by casting each gradient
tensor to ``float64`` *before* squaring -- this recovers the true
large-but-finite magnitude in the STATE B case (float64's representable
max, ``~1.8e308``, comfortably covers even ``(3.4e38)**2 ~= 1.16e77`` per
element), and is reported alongside the ordinary norm so a STATE B event
is never mistaken for "no information available" about its true severity.

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
    """Result of one guarded optimizer step attempt.

    ``gradient_state`` is one of ``"healthy"`` (STATE A), ``"norm_overflow"``
    (STATE B), or ``"element_nonfinite"`` (STATE C) -- see the module
    docstring. ``norm_overflow`` is a convenience boolean mirroring
    ``gradient_state == "norm_overflow"``."""

    applied: bool                              # True iff optimizer.step() was actually called
    grad_norm_pre_clip: float                  # ordinary (float32-native) norm; may be inf/nan
    grad_nonfinite_pre_clip: bool               # STATE C: at least one gradient ELEMENT is non-finite
    grad_nonfinite_post_clip: bool
    gradient_state: str = "healthy"             # "healthy" | "norm_overflow" | "element_nonfinite"
    norm_overflow: bool = False                 # STATE B: every element finite, aggregate norm overflowed
    grad_norm_robust: float = math.nan          # float64-accumulated diagnostic norm; nan if STATE C
    robust_norm_also_overflowed: bool = False   # honest flag: even the float64 diagnostic overflowed
    params_nonfinite_after_step: bool = False
    optimizer_state_nonfinite_after_step: bool = False

    @property
    def skipped_reason(self) -> str | None:
        if self.applied:
            return None
        if self.grad_nonfinite_pre_clip:
            return "nonfinite_gradient_pre_clip"
        if self.norm_overflow:
            return "norm_overflow"
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
    per-parameter norms, square-rooted, in the gradient's OWN (native,
    typically float32) dtype throughout -- the same combination PyTorch's
    own ``clip_grad_norm_`` computes internally. Deliberately does NOT
    special-case inf/nan -- callers want the true diagnostic value (inf/nan
    is itself meaningful signal, i.e. STATE B), only the *decision* to step
    is gated elsewhere. **This is the "ordinary" norm -- it CAN overflow
    even when every gradient element is individually finite** (squaring an
    element as small as ``~1.9e19`` already exceeds float32's representable
    max); see ``_grad_norm_robust_float64`` for the overflow-resistant
    diagnostic."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return math.sqrt(total)


def _grad_norm_robust_float64(params: list[torch.nn.Parameter]) -> float:
    """Total L2 gradient norm computed with float64 accumulation
    THROUGHOUT -- critically, each gradient tensor is upcast to
    ``float64`` **before** squaring, not just when combining already-
    computed (potentially already-overflowed) per-tensor norms. Squaring a
    float32-range value (up to ``~3.4e38``) in float32 arithmetic can
    itself overflow (``(3.4e38)**2 ~= 1.16e77``, far past float32's own
    max), even though the raw element is perfectly finite; float64 has
    ample headroom (max ``~1.8e308``) for this. Only call when every
    gradient element is already known finite (STATE A/B) -- an element
    that is itself non-finite makes this diagnostic meaningless (``nan``
    propagates through squaring regardless of dtype)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq_sum = torch.sum(g.to(torch.float64) ** 2)
            total += float(sq_sum.item())
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

    Guarantees: ``optimizer.step()`` is never called with a STATE-C
    (element-non-finite) gradient, NOR with a STATE-B (norm-overflow)
    gradient -- both are skipped and reported distinctly (see the module
    docstring). Parameters and optimizer state are checked finite after
    any step that IS applied; a caller observing ``outcome.collapsed ==
    True`` must fail training immediately (see docs/perception/
    kalmannet_training_instability_v1.md "Escalation Policy") -- this
    module does not itself abort training, it only reports the state so
    the caller's escalation policy can decide.
    """
    params = [p for p in net.parameters() if p.requires_grad]

    # STATE C check FIRST: at least one gradient ELEMENT is itself
    # non-finite. This must be checked independent of (and before) any
    # aggregate-norm computation, since a single non-finite element can
    # make the ordinary norm look non-finite too, but the converse
    # (STATE B) requires every element to be individually finite.
    grad_nonfinite_pre_clip = not _grads_finite(params)
    if grad_nonfinite_pre_clip:
        grad_norm_pre_clip = _grad_norm_safe(params)  # diagnostic only; will itself be inf/nan
        opt.zero_grad(set_to_none=False)
        return SafeStepOutcome(
            applied=False, grad_norm_pre_clip=grad_norm_pre_clip,
            grad_nonfinite_pre_clip=True, grad_nonfinite_post_clip=False,
            gradient_state="element_nonfinite", norm_overflow=False,
            grad_norm_robust=math.nan,
        )

    # Every element is finite -- now check whether the aggregate norm
    # REDUCTION itself overflowed (STATE B) before touching clip_grad_norm_.
    grad_norm_pre_clip = _grad_norm_safe(params)
    grad_norm_robust = _grad_norm_robust_float64(params)
    norm_overflow = not math.isfinite(grad_norm_pre_clip)

    if norm_overflow:
        # STATE B: do NOT call clip_grad_norm_ here. If we did, PyTorch
        # would compute clip_coef = max_norm / (inf + eps) == 0 and
        # multiply every still-finite gradient element by that zero --
        # 0 * finite == 0 exactly (safe, unlike STATE C's 0 * inf == nan),
        # but this would be an invisible, unreported effective no-op step.
        # Skip explicitly and report the event instead.
        opt.zero_grad(set_to_none=False)
        return SafeStepOutcome(
            applied=False, grad_norm_pre_clip=grad_norm_pre_clip,
            grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
            gradient_state="norm_overflow", norm_overflow=True,
            grad_norm_robust=grad_norm_robust,
            robust_norm_also_overflowed=not math.isfinite(grad_norm_robust),
        )

    # STATE A: healthy -- proceed with ordinary clipping + step.
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)

    grad_nonfinite_post_clip = not _grads_finite(params)
    if grad_nonfinite_post_clip:
        # Defensive: unreachable in practice now (both STATE B and STATE C
        # are already routed away above; clipping an already-finite,
        # finite-norm gradient set only ever scales it down or leaves it
        # unchanged), but checked explicitly per this task's own
        # instruction rather than assumed.
        opt.zero_grad(set_to_none=False)
        return SafeStepOutcome(
            applied=False, grad_norm_pre_clip=grad_norm_pre_clip,
            grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=True,
            gradient_state="element_nonfinite", norm_overflow=False,
            grad_norm_robust=grad_norm_robust,
        )

    opt.step()

    params_finite_after = _all_finite(params)
    opt_state_finite_after = optimizer_state_finite(opt)

    return SafeStepOutcome(
        applied=True, grad_norm_pre_clip=grad_norm_pre_clip,
        grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
        gradient_state="healthy", norm_overflow=False,
        grad_norm_robust=grad_norm_robust,
        params_nonfinite_after_step=not params_finite_after,
        optimizer_state_nonfinite_after_step=not opt_state_finite_after,
    )
