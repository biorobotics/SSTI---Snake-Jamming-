"""Regression checks for the original equations, signs and tracking semantics."""

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from anchoring_models.dynamics import forward_dynamics, mass_matrix, newton_euler
from anchoring_models.replay import replay
from anchoring_models.statics import Geometry, solve_initial
from anchoring_models.tracking import ContactTracker, Redistribution


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = Geometry.from_configuration(0.7245633602142334)
        cls.initial = solve_initial(cls.geometry, -2.6147518)

    def test_native_minmax_matches_saved_initial(self):
        np.testing.assert_allclose(self.initial.fn_native_N,
            [22.287063125953544, 44.57412625190709, 22.287063125953544], atol=2e-5)
        self.assertLess(self.initial.max_force_residual_N, 1e-6)
        self.assertLess(self.initial.max_moment_residual_Nm, 1e-6)
        self.assertLess(self.initial.max_contact_violation_N, 1e-6)

    def test_fn_only_is_postprocessing(self):
        native = self.initial.ft_native_N
        self.assertGreater(max(abs(v) for v in native), 4)
        output = ContactTracker(self.initial).update(0, 0, 0)
        np.testing.assert_array_equal(output["Ft_common_N"], [0, 0, 0])
        self.assertEqual(native, self.initial.ft_native_N)

    def test_contact1_transfer_and_weighted_sharing(self):
        state = ContactTracker(self.initial).update(0, 10, 5)
        base = np.asarray(self.initial.fn_native_N)
        np.testing.assert_allclose(state["Fn_N"], base+[-5, 5, 5])
        n = np.asarray(state["Fn_N"])
        np.testing.assert_allclose(state["Ft_common_N"], [-5*n[0]/(n[0]+n[2]), -5, -5*n[2]/(n[0]+n[2])])
        np.testing.assert_allclose(state["global_force_residual_N"], 0, atol=1e-6)

    def test_legacy_site_linear_and_contact2_sign(self):
        for site, index in [("Rpre1", 0), ("Rnext2", 2)]:
            state = ContactTracker(self.initial, Redistribution("legacy_site_linear", site)).update(0, 8, 3)
            expected = np.asarray(self.initial.fn_native_N).copy()
            expected[index] -= 3
            np.testing.assert_allclose(state["Fn_N"], expected)
            np.testing.assert_allclose(state["Ft_common_N"], [-2, -4, -2])
            np.testing.assert_allclose(state["global_force_residual_N"], 0, atol=1e-6)

    def test_friction_warning(self):
        output = ContactTracker(self.initial).update(0, 100, 0)
        self.assertTrue(output["warning_instantaneous"])
        self.assertIn("OUT_OF_CONE", output["contact_status"])

    def test_no_clipping_and_alarm_latches(self):
        tracker = ContactTracker(self.initial)
        loss = tracker.update(1, 40, 40)
        self.assertLess(loss["Fn_N"][0], 0)
        self.assertIsNone(loss["utilization"][0])
        recovered = tracker.update(2, 0, 0)
        self.assertFalse(recovered["warning_instantaneous"])
        self.assertTrue(recovered["warning_latched"])
        self.assertEqual(recovered["first_alarm_s"], 1)

    def test_snapshots_do_not_accumulate_and_moment_is_explicit(self):
        tracker = ContactTracker(self.initial)
        first = tracker.update(0, 4, 3, 2)
        second = tracker.update(1, 4, 3, 2)
        self.assertEqual(first["Fn_N"], second["Fn_N"])
        self.assertEqual(first["Ft_common_N"], second["Ft_common_N"])
        self.assertEqual(second["unmapped_applied_moment_Nm"], 2)
        self.assertFalse(second["full_two_body_moment_equilibrium_verified"])

    def test_invalid_sample_does_not_change_tracking_state(self):
        tracker = ContactTracker(self.initial)
        tracker.update(0, 0, 0)
        with self.assertRaises(ValueError):
            tracker.update(1, np.nan, 0)
        result = tracker.update(1, 0, 0)
        self.assertFalse(result["warning_latched"])
        with self.assertRaises(ValueError):
            tracker.update(1, 0, 0)

    def test_weighted_zero_denominator_and_invalid_configuration(self):
        empty = replace(self.initial, fn_native_N=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            ContactTracker(empty).update(0, 0, 0)
        with self.assertRaises(ValueError):
            Redistribution("contact1_transfer_weighted", "Rnext2")
        with self.assertRaises(ValueError):
            Geometry.from_configuration(float("nan"))

    def test_replay_preserves_samples(self):
        data = pd.DataFrame({"time_s": [0, .011, .024], "Fh_N": [0, 2, 4],
                             "Fv_N": [0, 1, 2], "measured_Fn1_N": [30, 29, 28]})
        output, _ = replay(data, self.initial, Redistribution())
        for column in data:
            np.testing.assert_array_equal(output[column], data[column])
        with self.assertRaises(ValueError):
            replay(data.iloc[[0, 0]], self.initial, Redistribution())


class DynamicsTests(unittest.TestCase):
    def test_one_rod_analytic(self):
        tau, wrench = newton_euler([0], [0], [1], [1], [1])
        np.testing.assert_allclose(tau, [[1/3]], atol=1e-14)
        np.testing.assert_allclose(wrench, [[0, -.5, -1/3]], atol=1e-14)

    def test_forward_inverse_closure_and_positive_mass(self):
        rng = np.random.default_rng(20261115)
        q, dq, ddq = [rng.normal(size=(12, 6)) for _ in range(3)]
        tau, wrench = newton_euler(q, dq, ddq)
        result = forward_dynamics(q, dq, tau)
        np.testing.assert_allclose(result["ddq_model_rad_s2"], ddq, atol=2e-11)
        np.testing.assert_allclose(result["robot_on_base_wrench"], wrench, atol=2e-12)
        np.testing.assert_allclose(wrench[:, 2], -tau[:, 0], atol=2e-12)
        self.assertTrue((np.linalg.eigvalsh(mass_matrix(q)) > 0).all())

    def test_same_mask_on_all_inputs_and_yaw(self):
        rng = np.random.default_rng(4)
        q, dq, tau = [rng.normal(size=(3, 6)) for _ in range(3)]
        signs = np.array([-1, 1, -1, 1, -1, 1])
        expected = forward_dynamics(q*signs, dq*signs, tau*signs)
        result = forward_dynamics(q, dq, tau, sign_mask=signs)
        np.testing.assert_allclose(result["robot_on_base_wrench"], expected["robot_on_base_wrench"])
        rotated = forward_dynamics(q, dq, tau, yaw_rad=np.pi/2)
        original = forward_dynamics(q, dq, tau)["robot_on_base_wrench"]
        np.testing.assert_allclose(rotated["robot_on_base_wrench"][:, 0], -original[:, 1], atol=1e-12)
        np.testing.assert_allclose(rotated["robot_on_base_wrench"][:, 1], original[:, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
