"""Real unit tests for inverse mapping, sampling, masks, and finite output bounds."""
import unittest
import numpy as np

try:
    from warp import warpImageNearestNeighbor, warpImageBilinear
except ModuleNotFoundError:
    warpImageNearestNeighbor = warpImageBilinear = None


class WarpTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(warpImageNearestNeighbor), 'A.3 inverse warping is not implemented')
        self.methods = (warpImageNearestNeighbor, warpImageBilinear)

    def test_identity_preserves_rgb_and_valid_black_pixels(self):
        image = np.array([[[0, 0, 0], [10, 50, 90]], [[255, 0, 3], [7, 8, 9]]], np.uint8)
        for method in self.methods:
            r = method(image, np.eye(3))
            np.testing.assert_array_equal(r.image, image)
            self.assertTrue(r.alpha.all())
            self.assertEqual(r.origin, (0, 0))

    def test_integer_translation_preserves_pixels_and_world_origin(self):
        image = np.arange(12).reshape(3, 4)
        H = np.array([[1., 0, -2], [0, 1, 3], [0, 0, 1]])
        for method in self.methods:
            r = method(image, H)
            self.assertEqual(r.origin, (-2, 3))
            np.testing.assert_array_equal(r.image, image)
            self.assertTrue(r.alpha.all())

    def test_half_pixel_bilinear_uses_four_independent_weights(self):
        image = np.array([[0., 10.], [20., 30.]])
        H = np.array([[1., 0, .5], [0, 1, .5], [0, 0, 1]])
        r = warpImageBilinear(image, H, canvas=(0, 0, 2, 2))
        np.testing.assert_array_equal(r.alpha, [[False, False], [False, True]])
        self.assertAlmostEqual(float(r.image[1, 1]), 15.)

    def test_nearest_half_ties_round_up(self):
        H = np.array([[1., 0, .5], [0, 1, .5], [0, 0, 1]])
        r = warpImageNearestNeighbor(np.array([[0., 10.], [20., 30.]]), H, canvas=(0, 0, 2, 2))
        self.assertEqual(float(r.image[1, 1]), 30.)

    def test_bilinear_ramp_at_quarter_pixel(self):
        H = np.array([[1., 0, .75], [0, 1, .75], [0, 0, 1]])
        r = warpImageBilinear(np.array([[0., 10.], [20., 30.]]), H, canvas=(1, 1, 1, 1))
        self.assertAlmostEqual(float(r.image[0, 0]), 7.5)

    def test_single_pixel_and_exact_last_row_column_are_valid(self):
        for method in self.methods:
            r = method(np.array([[42.]]), np.eye(3))
            self.assertEqual(float(r.image[0, 0]), 42.)
            self.assertTrue(r.alpha[0, 0])

    def test_inverse_perspective_mapping_matches_analytic_linear_ramp(self):
        image = np.arange(4)[None, :] + 10*np.arange(4)[:, None]
        H = np.array([[1., 0, 0], [0, 1, 0], [.1, 0, 1]])
        r = warpImageBilinear(image, H, canvas=(0, 0, 3, 3))
        self.assertAlmostEqual(float(r.image[1, 1]), 110/9, places=5)

    def test_fixed_canvas_outside_source_is_transparent(self):
        for method in self.methods:
            r = method(np.ones((2, 2)), np.eye(3), canvas=(3, 4, 2, 2))
            self.assertFalse(r.alpha.any())
            np.testing.assert_array_equal(r.image, 0.)

    def test_result_is_invariant_to_homogeneous_matrix_scale(self):
        image = np.arange(12).reshape(3, 4)
        for method in self.methods:
            a = method(image, np.eye(3))
            b = method(image, -1e-25*np.eye(3))
            np.testing.assert_array_equal(a.image, b.image)
            np.testing.assert_array_equal(a.alpha, b.alpha)

    def test_corner_center_bounds_include_all_transformed_corners(self):
        H = np.array([[1., 0, .2], [0, 1, -.3], [0, 0, 1]])
        for method in self.methods:
            r = method(np.ones((2, 3)), H)
            self.assertEqual(r.origin, (0, -1))
            self.assertEqual(r.image.shape, (3, 4))

    def test_automatic_bounds_reject_projective_horizon_crossing(self):
        H = np.array([[1., 0, 0], [0, 1, 0], [1, 0, -1]])
        for method in self.methods:
            with self.assertRaises(ValueError):
                method(np.ones((3, 3)), H)

    def test_rejects_oversized_canvas_before_allocation(self):
        for method in self.methods:
            with self.assertRaises(ValueError):
                method(np.ones((2, 2)), np.eye(3), canvas=(0, 0, 10000, 10000), max_pixels=100)

    def test_rejects_invalid_images_transforms_and_dimensions(self):
        for method in self.methods:
            for image, H, kwargs in [(np.ones((2, 2)), np.zeros((3, 3)), {}),
                                     (np.ones((2, 2)), np.full((3, 3), np.nan), {}),
                                     (np.zeros((0, 3)), np.eye(3), {}),
                                     (np.ones((2, 2)), np.eye(3), {'canvas':(0, 0, 0, 1)}),
                                     (np.ones((2, 2)), np.eye(3), {'block_rows':0})]:
                with self.subTest(), self.assertRaises(ValueError):
                    method(image, H, **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
