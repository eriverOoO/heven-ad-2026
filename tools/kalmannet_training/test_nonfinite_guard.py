"""Regression tests for the safe optimizer-step guard
(``nonfinite_guard.py``), reproducing the exact real-world dangerous case
found in the AV2 10k GENERIC-ROBUST run: ``clip_grad_norm_`` does not
neutralize a non-finite gradient, and Adam permanently poisons its
moment buffers if fed one directly."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nonfinite_guard import (  # noqa: E402
    optimizer_state_finite,
    safe_clip_and_step,
)


def _linear_with_adam(lr=0.004):
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    return net, opt


def test_clip_grad_norm_on_inf_gradient_produces_nan_not_zero():
    """Direct proof of the dangerous library behavior this guard exists
    to work around: clip_grad_norm_ does NOT safely zero an inf
    gradient -- it produces NaN (0 * inf == nan)."""
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e20
    loss = (net(x) ** 2).mean()
    assert not torch.isfinite(loss)  # this specific construction overflows the loss itself to inf
    loss.backward()
    assert not torch.isfinite(net.weight.grad).all()

    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    # The "clipped" gradient is NOT a safe zero -- it is NaN.
    assert torch.isnan(net.weight.grad).any()


def test_clip_grad_norm_on_directly_injected_nan_gradient_poisons_every_parameter_group():
    """Even MORE dangerous than the inf case: a NaN gradient in ONE
    parameter poisons clip_grad_norm_'s single shared total-norm/clip-
    coefficient computation, which then corrupts EVERY other parameter
    group's gradient too -- even one that started perfectly finite (here,
    an all-zero bias gradient). This is the exact 'nan gradient' case
    this task's own section 15 asks to verify explicitly, alongside the
    'finite large' and 'inf' cases above/below."""
    net = torch.nn.Linear(4, 4)
    net.weight.grad = torch.full_like(net.weight, float("nan"))
    net.bias.grad = torch.zeros_like(net.bias)  # perfectly finite, unrelated parameter

    returned_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    assert torch.isnan(returned_norm)
    assert torch.isnan(net.weight.grad).all()
    # The bias gradient was completely finite (all zeros) before clipping
    # and becomes NaN anyway -- clip_grad_norm_'s shared clip coefficient
    # poisons parameters that never had a bad gradient of their own.
    assert torch.isnan(net.bias.grad).all()


def test_finite_large_gradient_is_clipped_and_applied_normally():
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e9  # large but finite loss AND finite gradient
    loss = (net(x) ** 2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(net.weight.grad).all()

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.applied is True
    assert outcome.grad_nonfinite_pre_clip is False
    assert outcome.collapsed is False
    assert torch.isfinite(net.weight).all()
    assert optimizer_state_finite(opt)


def test_safe_step_skips_inf_gradient_without_touching_optimizer():
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e20
    loss = (net(x) ** 2).mean()
    loss.backward()
    assert not torch.isfinite(net.weight.grad).all()

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.applied is False
    assert outcome.grad_nonfinite_pre_clip is True
    assert outcome.collapsed is False
    # Parameters and optimizer state must be UNTOUCHED -- opt.step() was
    # never called.
    assert torch.isfinite(net.weight).all()
    assert len(opt.state) == 0  # Adam has not initialized state for any param yet


def test_safe_step_skips_nan_gradient_directly():
    net, opt = _linear_with_adam()
    net.zero_grad()
    net.weight.grad = torch.full_like(net.weight, float("nan"))
    net.bias.grad = torch.zeros_like(net.bias)

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.applied is False
    assert outcome.grad_nonfinite_pre_clip is True
    assert torch.isfinite(net.weight).all()


def test_adam_state_is_permanently_poisoned_by_one_nonfinite_gradient_if_guard_bypassed():
    """Demonstrates WHY the guard is necessary: calling the library
    functions directly (bypassing safe_clip_and_step) with one inf
    gradient permanently corrupts Adam state, and 5 subsequent fully
    finite gradient steps do NOT recover it."""
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e20
    loss = (net(x) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()  # the UNSAFE path -- exactly what the historical trainers did

    assert not torch.isfinite(net.weight).all()
    assert not optimizer_state_finite(opt)

    for _ in range(5):
        opt.zero_grad()
        xx = torch.randn(1, 4)
        yy = (net(xx) ** 2).mean()
        yy.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        # No recovery -- once poisoned, always poisoned.
        assert not torch.isfinite(net.weight).all()
        assert not optimizer_state_finite(opt)


def test_safe_step_prevents_the_permanent_poisoning_end_to_end():
    """The guard's whole point: the SAME inf-gradient event, routed
    through safe_clip_and_step, followed by 5 finite steps, leaves the
    model perfectly healthy throughout."""
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e20
    loss = (net(x) ** 2).mean()
    loss.backward()
    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.applied is False
    assert torch.isfinite(net.weight).all()

    for _ in range(5):
        opt.zero_grad()
        xx = torch.randn(1, 4)
        yy = (net(xx) ** 2).mean()
        yy.backward()
        outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
        assert outcome.applied is True
        assert outcome.collapsed is False
        assert torch.isfinite(net.weight).all()
        assert optimizer_state_finite(opt)


def test_optimizer_state_finite_true_for_fresh_optimizer():
    net, opt = _linear_with_adam()
    assert optimizer_state_finite(opt) is True


def test_optimizer_state_finite_detects_poisoned_state():
    net, opt = _linear_with_adam()
    xx = torch.randn(1, 4)
    (net(xx) ** 2).mean().backward()
    opt.step()
    assert optimizer_state_finite(opt) is True
    # Manually poison one moment buffer -- simulates a would-be corrupted state.
    state = next(iter(opt.state.values()))
    state["exp_avg"][0] = float("nan")
    assert optimizer_state_finite(opt) is False


def test_outcome_skipped_reason_reporting():
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e20
    (net(x) ** 2).mean().backward()
    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.skipped_reason == "nonfinite_gradient_pre_clip"

    net2, opt2 = _linear_with_adam()
    xx = torch.randn(1, 4)
    (net2(xx) ** 2).mean().backward()
    applied_outcome = safe_clip_and_step(net2, opt2, grad_clip=10.0)
    assert applied_outcome.skipped_reason is None


def test_grad_clip_none_still_guards_against_nonfinite_gradient():
    """grad_clip=None must not bypass the finiteness guard."""
    net, opt = _linear_with_adam()
    x = torch.ones(1, 4) * 1e20
    (net(x) ** 2).mean().backward()
    outcome = safe_clip_and_step(net, opt, grad_clip=None)
    assert outcome.applied is False
    assert torch.isfinite(net.weight).all()


# ---------------------------------------------------------------------------
# Three-state classification (see docs/perception/
# kalmannet_gradient_norm_overflow_v1.md): STATE A (healthy), STATE B
# (aggregate norm overflow, every element finite), STATE C (element
# non-finite). Section 6's four exact numerical cases: A, B, C, D.
# ---------------------------------------------------------------------------


def test_state_a_ordinary_finite_gradients_and_finite_norm():
    """Case A: ordinary finite gradients + finite norm -- healthy step."""
    net, opt = _linear_with_adam()
    xx = torch.randn(1, 4)
    (net(xx) ** 2).mean().backward()
    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.gradient_state == "healthy"
    assert outcome.norm_overflow is False
    assert outcome.grad_nonfinite_pre_clip is False
    assert outcome.applied is True
    assert math.isfinite(outcome.grad_norm_pre_clip)
    assert math.isfinite(outcome.grad_norm_robust)
    assert outcome.robust_norm_also_overflowed is False


def test_state_b_finite_elements_whose_float32_aggregate_norm_overflows():
    """Case B: every individual gradient element is finite (well within
    float32's representable range, ~3.4e38), but the ordinary (float32-
    native) L2-norm REDUCTION overflows to inf because squaring an
    element as small as ~1.9e19 already exceeds float32's max. Empirically
    verified threshold (see task session): a 4x4 tensor filled with 1e19
    triggers this deterministically."""
    net, opt = _linear_with_adam()
    net.zero_grad()
    huge_but_finite = 1e19
    net.weight.grad = torch.full_like(net.weight, huge_but_finite)
    net.bias.grad = torch.full_like(net.bias, huge_but_finite)

    # Prove every element is individually finite BEFORE calling the guard.
    assert torch.isfinite(net.weight.grad).all()
    assert torch.isfinite(net.bias.grad).all()
    # Prove the ordinary float32 norm reduction itself overflows.
    assert not math.isfinite(float(net.weight.grad.norm(2).item()))

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)

    # Classified as STATE B, never STATE C.
    assert outcome.gradient_state == "norm_overflow"
    assert outcome.norm_overflow is True
    assert outcome.grad_nonfinite_pre_clip is False
    assert outcome.applied is False
    # Ordinary norm reports inf (the defect this task fixes visibility for).
    assert not math.isfinite(outcome.grad_norm_pre_clip)
    # The robust float64 diagnostic recognizes the true large FINITE
    # magnitude instead of also reporting inf/nan.
    assert math.isfinite(outcome.grad_norm_robust)
    assert outcome.grad_norm_robust > 1e19
    assert outcome.robust_norm_also_overflowed is False
    # Optimizer state / parameters must be completely untouched -- no
    # step was ever attempted.
    assert torch.isfinite(net.weight).all()
    assert len(opt.state) == 0
    assert outcome.skipped_reason == "norm_overflow"


def test_state_b_reported_via_grad_norm_safe_matches_ordinary_norm():
    """The ordinary norm the guard reports for a STATE-B event is exactly
    what a plain (non-upcast) per-tensor float32 norm combination would
    give -- confirms the guard isn't silently using the robust value in
    place of the ordinary one."""
    net, opt = _linear_with_adam()
    net.zero_grad()
    net.weight.grad = torch.full_like(net.weight, 1e19)
    net.bias.grad = torch.zeros_like(net.bias)
    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.gradient_state == "norm_overflow"
    assert outcome.grad_norm_pre_clip == float("inf")


def test_state_c_one_gradient_element_is_inf():
    """Case C: at least one gradient element is inf -- must classify as
    element_nonfinite (STATE C), not norm_overflow, and skip exactly like
    the pre-existing behavior."""
    net, opt = _linear_with_adam()
    net.zero_grad()
    net.weight.grad = torch.full_like(net.weight, float("inf"))
    net.bias.grad = torch.zeros_like(net.bias)

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.gradient_state == "element_nonfinite"
    assert outcome.norm_overflow is False
    assert outcome.grad_nonfinite_pre_clip is True
    assert outcome.applied is False
    assert torch.isfinite(net.weight).all()
    assert len(opt.state) == 0
    assert outcome.skipped_reason == "nonfinite_gradient_pre_clip"


def test_state_c_one_gradient_element_is_nan():
    """Case D: one gradient element is nan -- same classification/skip as
    the inf case (both are STATE C)."""
    net, opt = _linear_with_adam()
    net.zero_grad()
    net.weight.grad = torch.full_like(net.weight, float("nan"))
    net.bias.grad = torch.zeros_like(net.bias)

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.gradient_state == "element_nonfinite"
    assert outcome.norm_overflow is False
    assert outcome.grad_nonfinite_pre_clip is True
    assert outcome.applied is False
    assert torch.isfinite(net.weight).all()
    assert len(opt.state) == 0


def test_state_b_never_calls_clip_grad_norm_and_reports_event_before_zeroing():
    """Directly proves the danger this task's policy avoids: if
    clip_grad_norm_ WERE called on a STATE-B gradient, it would silently
    zero it (0 * finite == 0) with NO event ever recorded. This guard
    skips BEFORE that call, reports the event with the TRUE (robust)
    pre-zero magnitude in ``outcome.grad_norm_robust``, and only THEN
    zeroes the gradient (per section 3's own "skip the optimizer step ...
    zero gradients" policy) -- the same explicit-zero-after-skip
    convention already used for STATE C, just now also visible via a
    dedicated, correctly-classified event rather than hiding inside an
    apparently-successful clipped step."""
    net, opt = _linear_with_adam()
    net.zero_grad()
    huge = 1e19
    net.weight.grad = torch.full_like(net.weight, huge)
    net.bias.grad = torch.full_like(net.bias, huge)

    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)

    # The event itself reports the TRUE robust magnitude, not a zero.
    assert outcome.gradient_state == "norm_overflow"
    assert outcome.grad_norm_robust > 1e19
    # Gradients are explicitly zeroed AFTER the event is reported (same
    # convention as STATE C) -- never left dangling for a caller to
    # accidentally step on.
    assert torch.equal(net.weight.grad, torch.zeros_like(net.weight))
    assert torch.equal(net.bias.grad, torch.zeros_like(net.bias))


def test_repeated_state_b_events_never_poison_adam_across_finite_steps():
    """A STATE-B event followed by ordinary finite steps must behave
    exactly like the STATE-C recovery test -- no lasting damage,
    regardless of how the event is classified."""
    net, opt = _linear_with_adam()
    net.zero_grad()
    net.weight.grad = torch.full_like(net.weight, 1e19)
    net.bias.grad = torch.zeros_like(net.bias)
    outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
    assert outcome.applied is False
    assert torch.isfinite(net.weight).all()

    for _ in range(5):
        opt.zero_grad()
        xx = torch.randn(1, 4)
        yy = (net(xx) ** 2).mean()
        yy.backward()
        outcome = safe_clip_and_step(net, opt, grad_clip=10.0)
        assert outcome.applied is True
        assert outcome.collapsed is False
        assert torch.isfinite(net.weight).all()
        assert optimizer_state_finite(opt)
