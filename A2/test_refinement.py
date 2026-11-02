"""Unit tests for the local, manually seeded annotation aid."""
import unittest
import numpy as np

try:
    from prepare_correspondences import refine_target
except ModuleNotFoundError:
    refine_target = None


class RefinementTests(unittest.TestCase):
    def test_recovers_known_local_translation_despite_brightness_change(self):
        self.assertTrue(callable(refine_target), "Local annotation refinement is not implemented")
        rng = np.random.default_rng(19)
        src = rng.uniform(10, 180, (140, 140)).astype(np.float32)
        dst = np.zeros_like(src)
        dst[6:, 14:] = src[:-6, :-14] * 1.2 + 10
        point, score, on_boundary = refine_target(src, dst, [50, 55], [70, 60], radius=12, half_patch=10)
        np.testing.assert_allclose(point, [64, 61], atol=.15)
        self.assertGreater(score, .999)
        self.assertFalse(on_boundary)

    def test_rejects_textureless_patch(self):
        self.assertTrue(callable(refine_target))
        with self.assertRaises(ValueError):
            refine_target(np.ones((140, 140), np.float32), np.ones((140, 140), np.float32),
                          [50, 55], [70, 60], radius=12, half_patch=10)

    def test_rejects_out_of_bounds_patch(self):
        self.assertTrue(callable(refine_target))
        with self.assertRaises(ValueError):
            refine_target(np.ones((140, 140), np.float32), np.ones((140, 140), np.float32),
                          [1, 1], [70, 60], radius=12, half_patch=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
