"""Regression tests for the safe optimizer-step guard
(``nonfinite_guard.py``), reproducing the exact real-world dangerous case
found in the AV2 10k GENERIC-ROBUST run: ``clip_grad_norm_`` does not
neutralize a non-finite gradient, and Adam permanently poisons its
moment buffers if fed one directly."""

from __future__ import annotations

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
