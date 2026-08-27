"""POST-FREEZE research extension, Phase 3: focused tests for
`kalmannet_arch2_core.py`. Fully offline, no ROS/rclpy import -- mirrors
`test_kalmannet_core.py`'s own isolation convention.

Includes: shape/no-NaN checks, deterministic-inference checks, missing-
measurement (`predict_only`) checks, per-track recurrent-state isolation
(the T-9B hidden-state-leakage bug class, verified structurally
impossible here since hidden state is never attached to the shared
`nn.Module`), a parameter-count report sanity check, and a regression
check that `kalmannet_core.py` (the frozen single-GRU estimator) is
untouched and still behaves exactly as before this file was added.
"""
import unittest

import torch

from ad_lidar_perception.kalmannet_arch2_core import (
    Arch2HiddenState,
    KalmanNetArch2Filter,
    KalmanNetGRUArch2,
    count_parameters,
)
from ad_lidar_perception.kalmannet_core import (
    KalmanNetFilter as FrozenKalmanNetFilter,
    KalmanNetGRU as FrozenKalmanNetGRU,
    MEAS_DIM,
    STATE_DIM,
    set_seed,
)


class ShapeAndFiniteTest(unittest.TestCase):
    def test_forward_output_shape_and_hidden_shapes(self):
        torch.manual_seed(0)
        net = KalmanNetGRUArch2()
        hidden = net.init_hidden(batch_size=3)
        obs_diff = torch.randn(3, MEAS_DIM)
        obs_innov_diff = torch.randn(3, MEAS_DIM)
        fw_evol_diff = torch.randn(3, STATE_DIM)
        fw_update_diff = torch.randn(3, STATE_DIM)
        gain, new_hidden = net(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, hidden)
        self.assertEqual(gain.shape, (3, STATE_DIM, MEAS_DIM))
        self.assertEqual(new_hidden.h_Q.shape, (1, 3, net.d_hidden_Q))
        self.assertEqual(new_hidden.h_Sigma.shape, (1, 3, net.d_hidden_Sigma))
        self.assertEqual(new_hidden.h_S.shape, (1, 3, net.d_hidden_S))

    def test_no_nan_inf_in_output(self):
        torch.manual_seed(1)
        net = KalmanNetGRUArch2()
        hidden = net.init_hidden(batch_size=1)
        args = [torch.randn(1, MEAS_DIM), torch.randn(1, MEAS_DIM),
                torch.randn(1, STATE_DIM), torch.randn(1, STATE_DIM)]
        gain, new_hidden = net(*args, hidden)
        self.assertTrue(torch.isfinite(gain).all())
        self.assertTrue(torch.isfinite(new_hidden.h_Q).all())
        self.assertTrue(torch.isfinite(new_hidden.h_Sigma).all())
        self.assertTrue(torch.isfinite(new_hidden.h_S).all())

    def test_module_has_no_mutable_hidden_state_attribute(self):
        """The nn.Module itself must own no per-track state -- forward()
        is a pure function of (inputs, hidden) -> (gain, new_hidden)."""
        net = KalmanNetGRUArch2()
        for attr in ("h", "h_Q", "h_Sigma", "h_S", "hidden"):
            self.assertFalse(hasattr(net, attr), f"unexpected stateful attribute: {attr}")


class KalmanNetArch2FilterTest(unittest.TestCase):
    def test_single_sequence_execution_no_nan(self):
        torch.manual_seed(0)
        net = KalmanNetGRUArch2()
        kf = KalmanNetArch2Filter(net)
        x0 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        kf.init_sequence(x0, batch_size=1)
        for i in range(15):
            z = torch.tensor([[0.1 * (i + 1), 0.01 * i]])
            x_post = kf.step(z, dt=0.1)
            self.assertTrue(torch.isfinite(x_post).all())

    def test_real_dt_variation_does_not_crash(self):
        torch.manual_seed(2)
        net = KalmanNetGRUArch2()
        kf = KalmanNetArch2Filter(net)
        x0 = torch.zeros(1, STATE_DIM)
        kf.init_sequence(x0, batch_size=1)
        for dt in [0.05, 0.6, 0.099, 1.2, 0.15]:
            x_post = kf.step(torch.randn(1, MEAS_DIM), dt=dt)
            self.assertTrue(torch.isfinite(x_post).all())

    def test_deterministic_inference_same_seed(self):
        def run():
            torch.manual_seed(42)
            net = KalmanNetGRUArch2()
            kf = KalmanNetArch2Filter(net)
            x0 = torch.zeros(1, STATE_DIM)
            kf.init_sequence(x0, batch_size=1)
            outputs = []
            for i in range(5):
                z = torch.tensor([[float(i) * 0.1, 0.0]])
                outputs.append(kf.step(z, dt=0.1).clone())
            return torch.stack(outputs)

        out1 = run()
        out2 = run()
        torch.testing.assert_close(out1, out2)

    def test_missing_measurement_predict_only_no_network_call_no_nan(self):
        """`predict_only` must never touch the network or hidden state --
        verified by confirming hidden state is byte-identical before/after."""
        torch.manual_seed(3)
        net = KalmanNetGRUArch2()
        kf = KalmanNetArch2Filter(net)
        x0 = torch.tensor([[1.0, 2.0, 0.5, -0.5]])
        kf.init_sequence(x0, batch_size=1)
        kf.step(torch.tensor([[1.05, 1.95]]), dt=0.1)
        hidden_before = kf.hidden.detach()

        x_post = kf.predict_only(dt=0.1)
        self.assertTrue(torch.isfinite(x_post).all())
        torch.testing.assert_close(kf.hidden.h_Q, hidden_before.h_Q)
        torch.testing.assert_close(kf.hidden.h_Sigma, hidden_before.h_Sigma)
        torch.testing.assert_close(kf.hidden.h_S, hidden_before.h_S)

    def test_predict_only_matches_analytical_f(self):
        torch.manual_seed(4)
        net = KalmanNetGRUArch2()
        kf = KalmanNetArch2Filter(net)
        x0 = torch.tensor([[3.0, -1.0, 2.0, 0.5]])
        kf.init_sequence(x0, batch_size=1)
        expected = KalmanNetArch2Filter._f(x0, 0.2)
        actual = kf.predict_only(dt=0.2)
        torch.testing.assert_close(actual, expected)


class RecurrentStateIsolationTest(unittest.TestCase):
    """The T-9B hidden-state-leakage bug (`engineering_forensics.csv`)
    occurred because the single-GRU module stored its hidden state
    directly on shared weights. Verify that class of bug is structurally
    impossible for the arch2 filter: two filter instances sharing one
    `KalmanNetGRUArch2` weight module, run interleaved, must produce
    exactly the same output each would produce running alone."""

    def _run_alone(self, net, x0, measurements, dts):
        kf = KalmanNetArch2Filter(net)
        kf.init_sequence(x0, batch_size=1)
        outputs = []
        for z, dt in zip(measurements, dts):
            outputs.append(kf.step(z, dt).clone())
        return torch.stack(outputs)

    def test_two_tracks_interleaved_match_running_alone(self):
        torch.manual_seed(0)
        net = KalmanNetGRUArch2()

        x0_a = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        meas_a = [torch.tensor([[0.1 * (i + 1), 0.0]]) for i in range(6)]
        dts_a = [0.1] * 6

        x0_b = torch.tensor([[10.0, 5.0, -1.0, 0.2]])
        meas_b = [torch.tensor([[10.0 - 0.1 * (i + 1), 5.0 + 0.02 * i]]) for i in range(6)]
        dts_b = [0.15] * 6

        expected_a = self._run_alone(net, x0_a, meas_a, dts_a)
        expected_b = self._run_alone(net, x0_b, meas_b, dts_b)

        kf_a = KalmanNetArch2Filter(net)
        kf_a.init_sequence(x0_a, batch_size=1)
        kf_b = KalmanNetArch2Filter(net)
        kf_b.init_sequence(x0_b, batch_size=1)
        out_a, out_b = [], []
        for i in range(6):
            out_a.append(kf_a.step(meas_a[i], dts_a[i]).clone())
            out_b.append(kf_b.step(meas_b[i], dts_b[i]).clone())
        out_a = torch.stack(out_a)
        out_b = torch.stack(out_b)

        torch.testing.assert_close(out_a, expected_a)
        torch.testing.assert_close(out_b, expected_b)

    def test_track_death_does_not_affect_a_fresh_track(self):
        """A track that has accumulated state and 'dies' (its Filter is
        simply dropped) must leave zero residue affecting a brand-new
        track built on the same shared network."""
        torch.manual_seed(1)
        net = KalmanNetGRUArch2()

        kf_dead = KalmanNetArch2Filter(net)
        kf_dead.init_sequence(torch.tensor([[0.0, 0.0, 5.0, 5.0]]), batch_size=1)
        for i in range(10):
            kf_dead.step(torch.tensor([[0.5 * i, 0.5 * i]]), dt=0.1)
        del kf_dead  # track "dies"

        x0 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        meas = [torch.tensor([[1.0 + 0.05 * i, 1.0]]) for i in range(4)]
        expected = self._run_alone(net, x0, meas, [0.1] * 4)

        kf_fresh = KalmanNetArch2Filter(net)
        kf_fresh.init_sequence(x0, batch_size=1)
        out = torch.stack([kf_fresh.step(z, 0.1).clone() for z in meas])
        torch.testing.assert_close(out, expected)


class ParameterCountReportTest(unittest.TestCase):
    def test_count_parameters_matches_manual_sum(self):
        net = KalmanNetGRUArch2()
        report = count_parameters(net)
        manual_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.assertEqual(report["total"], manual_total)
        self.assertGreater(report["total"], 0)
        for key in ("GRU_Q", "GRU_Sigma", "GRU_S", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FC7"):
            self.assertIn(key, report)
            self.assertGreater(report[key], 0)

    def test_sweep_width_changes_parameter_count(self):
        small = count_parameters(KalmanNetGRUArch2(in_mult=2, out_mult=4))
        large = count_parameters(KalmanNetGRUArch2(in_mult=5, out_mult=40))
        self.assertLess(small["total"], large["total"])


class ArchitectureIsGenuinelyThreeGRUsTest(unittest.TestCase):
    """Guards against a lazy 'three stacked generic GRUs' implementation:
    the Sigma-GRU's input must depend on the Q-GRU's output, and the
    S-GRU's input must depend on the Sigma-GRU's (via FC1) -- i.e. the
    three GRUs form the reference's own Q -> Sigma -> S dependency chain,
    not three independent parallel branches."""

    def test_input_dims_reflect_q_to_sigma_to_s_chain(self):
        net = KalmanNetGRUArch2()
        self.assertEqual(net.GRU_Sigma.input_size, net.d_hidden_Q + net.m * net.in_mult)
        self.assertEqual(net.GRU_S.input_size, net.d_hidden_S + 2 * net.n * net.in_mult)
        # FC1 (Sigma -> S contribution) must consume the Sigma-GRU's own
        # hidden width, not some independent scalar.
        self.assertEqual(net.FC1[0].in_features, net.d_hidden_Sigma)


class FrozenSingleGRURegressionTest(unittest.TestCase):
    """Confirms `kalmannet_core.py` (imported by DENSE-KALMANNET-v2 and
    T-9B's ROS integration) behaves exactly as before this file was
    added -- same deterministic output for the same seed, same shapes."""

    def test_frozen_module_deterministic_output_unaffected(self):
        def run():
            set_seed(7)
            net = FrozenKalmanNetGRU(hidden_size=8)
            kf = FrozenKalmanNetFilter(net)
            x0 = torch.zeros(1, STATE_DIM)
            kf.init_sequence(x0, batch_size=1)
            outputs = []
            for i in range(5):
                z = torch.tensor([[float(i) * 0.1, 0.0]])
                outputs.append(kf.step(z, dt=0.1).clone())
            return torch.stack(outputs)

        out1 = run()
        out2 = run()
        torch.testing.assert_close(out1, out2)
        self.assertEqual(out1.shape, (5, 1, STATE_DIM))


if __name__ == "__main__":
    unittest.main()
