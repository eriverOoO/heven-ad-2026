"""POST-FREEZE research extension, Phase 7: focused tests for
`kalmannet_uncertainty_core.py`. Offline, no ROS/rclpy import."""
import unittest

import torch

from ad_lidar_perception.kalmannet_uncertainty_core import (
    InnovationCovarianceHead,
    cholesky_to_cov,
    gaussian_nll,
)


class PositiveDefiniteGuaranteeTest(unittest.TestCase):
    def test_covariance_always_positive_definite(self):
        torch.manual_seed(0)
        head = InnovationCovarianceHead()
        for _ in range(20):
            args = [torch.randn(4, 2) * 100, torch.randn(4, 2) * 100,
                    torch.randn(4, 4) * 100, torch.randn(4, 4) * 100]
            L = head(*args)
            S = cholesky_to_cov(L)
            eigvals = torch.linalg.eigvalsh(S)
            self.assertTrue((eigvals > 0).all())

    def test_no_nan_inf_even_on_extreme_inputs(self):
        head = InnovationCovarianceHead()
        args = [torch.full((2, 2), 1e6), torch.full((2, 2), -1e6),
                torch.zeros(2, 4), torch.full((2, 4), 1e-8)]
        L = head(*args)
        self.assertTrue(torch.isfinite(L).all())
        S = cholesky_to_cov(L)
        self.assertTrue(torch.isfinite(S).all())


class GaussianNLLTest(unittest.TestCase):
    def test_nll_matches_closed_form_isotropic_case(self):
        # For S = sigma^2 * I (2D), NLL = ||y||^2/(2 sigma^2) + log(2*pi*sigma^2)
        sigma = 2.0
        L = torch.tensor([[[sigma, 0.0], [0.0, sigma]]])
        y = torch.tensor([[1.0, -1.5]])
        nll = gaussian_nll(y, L)
        expected = 0.5 * ((y ** 2).sum() / sigma ** 2 + 2 * torch.log(torch.tensor(sigma ** 2)) + 2 * torch.log(torch.tensor(2 * torch.pi)))
        torch.testing.assert_close(nll[0], expected, atol=1e-4, rtol=1e-4)

    def test_nll_finite_for_batch(self):
        head = InnovationCovarianceHead()
        args = [torch.randn(6, 2), torch.randn(6, 2), torch.randn(6, 4), torch.randn(6, 4)]
        L = head(*args)
        innov = torch.randn(6, 2)
        nll = gaussian_nll(innov, L)
        self.assertTrue(torch.isfinite(nll).all())
        self.assertEqual(nll.shape, (6,))

    def test_smaller_covariance_penalizes_large_residual_more(self):
        """Sanity check on the NLL's own direction: a small predicted S
        should give a HIGHER (worse) NLL than a large S when the actual
        residual is large -- confirms the loss can actually teach
        calibration, not just decrease trivially."""
        y = torch.tensor([[5.0, 5.0]])
        L_small = torch.tensor([[[0.1, 0.0], [0.0, 0.1]]])
        L_large = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
        nll_small = gaussian_nll(y, L_small)
        nll_large = gaussian_nll(y, L_large)
        self.assertGreater(nll_small.item(), nll_large.item())


class GradientFlowTest(unittest.TestCase):
    def test_gradients_reach_all_head_parameters(self):
        torch.manual_seed(1)
        head = InnovationCovarianceHead()
        args = [torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 4), torch.randn(3, 4)]
        L = head(*args)
        innov = torch.randn(3, 2)
        loss = gaussian_nll(innov, L).mean()
        loss.backward()
        for name, p in head.named_parameters():
            self.assertIsNotNone(p.grad, f"no gradient reached {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), f"non-finite gradient at {name}")


if __name__ == "__main__":
    unittest.main()
