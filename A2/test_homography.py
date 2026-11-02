"""Unit tests for the A.2 solver; no mocks or third-party test framework."""
import unittest
import numpy as np

try:
    from homography import computeH, project_points, build_system
except ModuleNotFoundError:
    computeH = project_points = build_system = None


class HomographyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(computeH), "A.2 computeH is not implemented")
        self.pts = np.array([[0, 0], [250, 0], [250, 180], [0, 180],
                             [30, 60], [170, 20], [80, 150], [210, 130]], dtype=float)
        self.H = np.array([[1.08, .06, 73], [-.03, .95, -21], [.0003, -.0002, 1.]])

    def target(self, pts):
        homogeneous = np.column_stack((pts, np.ones(len(pts)))) @ self.H.T
        return homogeneous[:, :2] / homogeneous[:, 2:3]

    def test_recovers_exact_four_point_projective_transform(self):
        result = computeH(self.pts[:4], self.target(self.pts[:4]))
        np.testing.assert_allclose(result, self.H, atol=1e-10)

    def test_recovers_overdetermined_transform_and_maps_first_to_second(self):
        target = self.target(self.pts)
        H = computeH(self.pts, target)
        np.testing.assert_allclose(project_points(self.pts, H), target, atol=1e-9)
        np.testing.assert_allclose(project_points(target, np.linalg.inv(H)), self.pts, atol=1e-9)

    def test_coordinate_normalization_handles_large_offsets(self):
        pts = self.pts + np.array([1e7, -2e7])
        target = pts @ np.array([[1.01, .04], [-.03, .96]]).T + [120, -87]
        H = computeH(pts, target)
        np.testing.assert_allclose(project_points(pts, H), target, atol=1e-5, rtol=0)

    def test_least_squares_predicts_held_out_noisy_points(self):
        rng = np.random.default_rng(671)
        pts = rng.uniform([0, 0], [3000, 4000], size=(50, 2))
        noisy = self.target(pts) + rng.normal(0, .25, (50, 2))
        H = computeH(pts, noisy)
        held_out = np.array([[300, 450], [2000, 1400], [1500, 3000]])
        errors = np.linalg.norm(project_points(held_out, H) - self.target(held_out), axis=1)
        self.assertLess(float(errors.max()), .5)

    def test_raw_equation_system_matches_known_transform(self):
        A, b = build_system(self.pts, self.target(self.pts))
        self.assertEqual(A.shape, (16, 8))
        self.assertEqual(b.shape, (16,))
        np.testing.assert_allclose(A @ self.H.ravel()[:8], b, atol=1e-10)

    def test_rejects_bad_shape_count_and_nonfinite_values(self):
        invalid = [(self.pts[:3], self.pts[:3]), (self.pts, self.pts[:4]),
                   (np.zeros((4, 3)), np.zeros((4, 3))),
                   (self.pts * np.nan, self.pts), (self.pts, self.pts * np.inf)]
        for src, dst in invalid:
            with self.subTest(shape=np.shape(src)), self.assertRaises(ValueError):
                computeH(src, dst)

    def test_rejects_collinear_or_repeated_points(self):
        line = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        for src, dst in [(line, line), (np.ones((5, 2)), np.ones((5, 2))),
                         (self.pts, np.column_stack((np.arange(8), np.zeros(8))))]:
            with self.subTest(), self.assertRaises(ValueError):
                computeH(src, dst)

    def test_projection_rejects_points_at_infinity(self):
        with self.assertRaises(ValueError):
            project_points(np.array([[0., 1.]]), np.array([[1., 0, 0], [0, 1, 0], [1, 0, 0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
