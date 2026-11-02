"""Tests for annotation coordinate conversion and differently scaled local patches."""
import unittest
import numpy as np
from PIL import Image
import prepare_correspondences as annotations


class MixedResolutionTests(unittest.TestCase):
    def test_preview_centers_map_to_original_centers_per_image(self):
        self.assertTrue(hasattr(annotations, 'preview_to_original'), 'Pixel-center conversion is not implemented')
        p = np.array([[0., 0.], [1007., 1343.], [503.5, 671.5]])
        small = annotations.preview_to_original(p, [3024, 4032])
        large = annotations.preview_to_original(p, [4284, 5712])
        np.testing.assert_allclose(small, [[1, 1], [3022, 4030], [1511.5, 2015.5]])
        np.testing.assert_allclose(large, [[1.625, 1.625], [4281.375, 5709.375], [2141.5, 2855.5]])

    def test_local_refinement_recovers_center_with_different_template_scale(self):
        self.assertTrue(hasattr(annotations, 'refine_scaled_target'), 'Scaled local annotation aid is not implemented')
        rng = np.random.default_rng(8443)
        src = rng.uniform(0, 200, (140, 140)).astype(np.float32)
        patch = src[40:71, 35:66]
        enlarged = np.asarray(Image.fromarray(patch).resize((47, 47), Image.Resampling.BICUBIC))
        target = np.zeros((160, 160), np.float32)
        target[60:107, 70:117] = enlarged * 1.1 + 8
        xy, score, boundary = annotations.refine_scaled_target(src, target, [50, 55], [90, 80], 1.5, radius=16)
        np.testing.assert_allclose(xy, [93, 83], atol=.2)
        self.assertGreater(score, .99)
        self.assertFalse(boundary)


if __name__ == '__main__':
    unittest.main(verbosity=2)
