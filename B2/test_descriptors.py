import unittest
import numpy as np
from descriptors import extract_descriptors

class DescriptorTests(unittest.TestCase):
    def test_constant_invalid(self):
        r=extract_descriptors(np.full((90,90),.4),np.array([[45,45.]]))
        self.assertFalse(r['valid'][0]);self.assertTrue(r['in_bounds'][0])
        np.testing.assert_array_equal(r['descriptors'],0)

    def test_linear_ramp_sampling_and_normalization(self):
        y,x=np.mgrid[:100,:100];gray=(2*x+3*y)/500
        r=extract_descriptors(gray,np.array([[50,50.]]))
        offsets=np.arange(8)*5-17.5
        expected=(2*(50+offsets[None,:])+3*(50+offsets[:,None]))/500
        np.testing.assert_allclose(r['sampled8'][0],expected,atol=1e-12)
        self.assertAlmostEqual(r['descriptors'][0].mean(),0,places=12)
        self.assertAlmostEqual(r['descriptors'][0].std(),1,places=12)
        self.assertEqual(r['descriptors'].shape,(1,64))

    def test_positive_gain_bias_invariance_without_clipping(self):
        y,x=np.mgrid[:100,:100];gray=np.sin(x*.2)+np.cos(y*.13)
        xy=np.array([[50,50.],[60,40.]])
        a=extract_descriptors(gray,xy);b=extract_descriptors(gray*2.7+4.1,xy)
        np.testing.assert_allclose(a['descriptors'],b['descriptors'],atol=1e-12)

    def test_invalid_border_preserves_rows(self):
        y,x=np.mgrid[:100,:100]
        r=extract_descriptors((x+y)/200,np.array([[2,2],[50,50],[99,99.]]))
        np.testing.assert_array_equal(r['valid'],[False,True,False])
        np.testing.assert_array_equal(r['descriptors'][[0,2]],0)
        self.assertEqual(r['sampled8'].shape,(3,8,8))

    def test_half_pixel_symmetric_raw_region(self):
        y,x=np.mgrid[:100,:100]
        r=extract_descriptors(x.astype(float),np.array([[50,50.]]))
        np.testing.assert_allclose(r['raw40'][0,0],[30.5+i for i in range(40)],atol=1e-12)

    def test_empty_and_invalid(self):
        r=extract_descriptors(np.ones((50,50)),np.empty((0,2)))
        self.assertEqual(r['descriptors'].shape,(0,64))
        with self.assertRaises(ValueError):extract_descriptors(np.ones((50,50)),np.array([[np.nan,2]]))
        with self.assertRaises(ValueError):extract_descriptors(np.full((50,50),np.nan),np.array([[25,25]]))

if __name__=='__main__':unittest.main()
