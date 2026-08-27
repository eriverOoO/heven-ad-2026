"""T-9A Phase 19: focused tests for kalmannet_core.py. Fully offline, no
ROS/rclpy import -- consistent with the module's own isolation."""
import math
import os
import tempfile
import unittest

import numpy as np
import torch

from ad_lidar_perception.kalmannet_core import (
    F_matrix, H_matrix, Q_matrix, LinearCVKF, KalmanNetGRU, KalmanNetFilter,
    set_seed, STATE_DIM, MEAS_DIM,
)


class FMatrixTest(unittest.TestCase):
    def test_identity_at_zero_dt(self):
        F = F_matrix(0.0)
        np.testing.assert_allclose(F, np.eye(4))

    def test_position_advances_by_velocity_times_dt(self):
        F = F_matrix(0.5)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        x_next = F @ x
        np.testing.assert_allclose(x_next, [1.0 + 1.5, 2.0 + 2.0, 3.0, 4.0])

    def test_real_variable_dt_supported(self):
        for dt in [0.05, 0.1, 0.099, 0.6, 1.2]:
            F = F_matrix(dt)
            self.assertAlmostEqual(F[0, 2], dt)
            self.assertAlmostEqual(F[1, 3], dt)


class MeasurementProjectionTest(unittest.TestCase):
    def test_H_projects_position_only(self):
        x = np.array([5.0, -3.0, 1.0, 2.0])
        z = H_matrix @ x
        np.testing.assert_allclose(z, [5.0, -3.0])

    def test_Q_positive_semi_definite(self):
        Q = Q_matrix(0.15, sigma_a=1.0)
        eigvals = np.linalg.eigvalsh(Q)
        self.assertTrue(np.all(eigvals >= -1e-9))


class LinearCVKFTest(unittest.TestCase):
    def test_predict_update_no_nan(self):
        kf = LinearCVKF()
        kf.init_sequence(np.array([0.0, 0.0, 1.0, 0.0]))
        for i in range(20):
            kf.predict(0.1)
            x = kf.update(np.array([0.1 * (i + 1), 0.0]))
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.isfinite(kf.P)))

    def test_converges_toward_constant_velocity_measurement(self):
        kf = LinearCVKF(sigma_a=0.5, r_std=0.05)
        kf.init_sequence(np.array([0.0, 0.0, 0.0, 0.0]))
        true_v = 2.0
        x = None
        for i in range(1, 60):
            kf.predict(0.1)
            z = np.array([true_v * 0.1 * i, 0.0])
            x = kf.update(z)
        self.assertAlmostEqual(x[2], true_v, delta=0.3)


class KalmanNetGRUShapeTest(unittest.TestCase):
    def test_forward_output_shape(self):
        net = KalmanNetGRU(hidden_size=8)
        net.reset_hidden(batch_size=3)
        obs_diff = torch.randn(3, MEAS_DIM)
        obs_innov_diff = torch.randn(3, MEAS_DIM)
        fw_evol_diff = torch.randn(3, STATE_DIM)
        fw_update_diff = torch.randn(3, STATE_DIM)
        K = net(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.assertEqual(K.shape, (3, STATE_DIM, MEAS_DIM))

    def test_no_nan_inf_in_output(self):
        net = KalmanNetGRU(hidden_size=8)
        net.reset_hidden(batch_size=1)
        args = [torch.randn(1, MEAS_DIM), torch.randn(1, MEAS_DIM),
                torch.randn(1, STATE_DIM), torch.randn(1, STATE_DIM)]
        K = net(*args)
        self.assertTrue(torch.isfinite(K).all())

    def test_recurrent_reset_changes_state(self):
        net = KalmanNetGRU(hidden_size=8)
        net.reset_hidden(batch_size=1)
        h_after_reset = net.h.clone()
        args = [torch.randn(1, MEAS_DIM), torch.randn(1, MEAS_DIM),
                torch.randn(1, STATE_DIM), torch.randn(1, STATE_DIM)]
        net(*args)
        self.assertFalse(torch.allclose(net.h, h_after_reset))
        net.reset_hidden(batch_size=1)
        self.assertTrue(torch.allclose(net.h, h_after_reset))


class KalmanNetFilterTest(unittest.TestCase):
    def test_single_sequence_execution_no_nan(self):
        set_seed(0)
        net = KalmanNetGRU(hidden_size=8)
        knf = KalmanNetFilter(net)
        x0 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        knf.init_sequence(x0, batch_size=1)
        for i in range(15):
            z = torch.tensor([[0.1 * (i + 1), 0.01 * i]])
            x_post = knf.step(z, dt=0.1)
            self.assertTrue(torch.isfinite(x_post).all())

    def test_batched_sequence_execution(self):
        set_seed(0)
        net = KalmanNetGRU(hidden_size=8)
        knf = KalmanNetFilter(net)
        x0 = torch.zeros(4, STATE_DIM)
        knf.init_sequence(x0, batch_size=4)
        for i in range(10):
            z = torch.randn(4, MEAS_DIM)
            x_post = knf.step(z, dt=0.15)
            self.assertEqual(x_post.shape, (4, STATE_DIM))
            self.assertTrue(torch.isfinite(x_post).all())

    def test_sequence_boundaries_do_not_leak_hidden_state(self):
        set_seed(1)
        net = KalmanNetGRU(hidden_size=8)
        knf = KalmanNetFilter(net)
        x0 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

        knf.init_sequence(x0, batch_size=1)
        for i in range(5):
            knf.step(torch.tensor([[float(i), 0.0]]), dt=0.1)
        h_after_seq1 = net.h.clone()

        knf.init_sequence(x0, batch_size=1)
        h_after_reset = net.h.clone()
        self.assertFalse(torch.allclose(h_after_seq1, h_after_reset))

    def test_real_dt_variation_does_not_crash(self):
        set_seed(2)
        net = KalmanNetGRU(hidden_size=8)
        knf = KalmanNetFilter(net)
        x0 = torch.zeros(1, STATE_DIM)
        knf.init_sequence(x0, batch_size=1)
        for dt in [0.05, 0.6, 0.099, 1.2, 0.15]:
            x_post = knf.step(torch.randn(1, MEAS_DIM), dt=dt)
            self.assertTrue(torch.isfinite(x_post).all())

    def test_deterministic_inference_same_seed(self):
        def run():
            set_seed(42)
            net = KalmanNetGRU(hidden_size=8)
            knf = KalmanNetFilter(net)
            x0 = torch.zeros(1, STATE_DIM)
            knf.init_sequence(x0, batch_size=1)
            outputs = []
            for i in range(5):
                z = torch.tensor([[float(i) * 0.1, 0.0]])
                outputs.append(knf.step(z, dt=0.1).clone())
            return torch.stack(outputs)

        out1 = run()
        out2 = run()
        torch.testing.assert_close(out1, out2)


class SaveLoadTest(unittest.TestCase):
    def test_checkpoint_reload_identical(self):
        set_seed(0)
        net = KalmanNetGRU(hidden_size=8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            torch.save(net.state_dict(), path)

            net2 = KalmanNetGRU(hidden_size=8)
            net2.load_state_dict(torch.load(path, weights_only=True))

            net.reset_hidden(1)
            net2.reset_hidden(1)
            args = [torch.randn(1, MEAS_DIM), torch.randn(1, MEAS_DIM),
                    torch.randn(1, STATE_DIM), torch.randn(1, STATE_DIM)]
            out1 = net(*[a.clone() for a in args])
            out2 = net2(*[a.clone() for a in args])
            torch.testing.assert_close(out1, out2)


if __name__ == "__main__":
    unittest.main()
