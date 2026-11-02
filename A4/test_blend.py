import unittest
import numpy as np
from blend import feather_weights, blend_layers

class BlendTests(unittest.TestCase):
    def test_known_weighted_average(self):
        a=np.full((2,3,3),20); b=np.full((2,3,3),100)
        out,mask=blend_layers([(a,np.ones((2,3))), (b,np.full((2,3),3))])
        np.testing.assert_array_equal(out,80); self.assertTrue(mask.all())

    def test_opaque_black_and_single_support(self):
        a=np.zeros((1,3,3)); b=np.full_like(a,100)
        out,mask=blend_layers([(a,np.array([[1,1,0]])), (b,np.array([[0,1,0]]))])
        np.testing.assert_array_equal(out[0,:,0],[0,50,0]); np.testing.assert_array_equal(mask,[[1,1,0]])

    def test_positive_symmetric_edge_weights(self):
        w=feather_weights((5,7))
        self.assertGreater(w.min(),0); self.assertEqual(w.max(),1)
        np.testing.assert_array_equal(w,w[::-1,::-1]); self.assertGreater(w[2,3],w[0,3])
        np.testing.assert_array_equal(feather_weights((1,1)),[[1]])

    def test_invalid_weights(self):
        for w in [np.array([[-1.]]),np.array([[np.nan]])]:
            with self.assertRaises(ValueError):blend_layers([(np.ones((1,1,3)),w)])
        with self.assertRaises(ValueError):blend_layers([])
        with self.assertRaises(ValueError):blend_layers([(np.ones((1,1,3)),np.ones((2,2)))])

    def test_single_support_fractional_values_are_exact(self):
        pixels=np.array([[[239.49998,244.50002,190.49998]]],np.float32)
        for weight in [0.0031,0.017,0.711]:
            result,_=blend_layers([(pixels,np.full((1,1),weight)),(np.zeros_like(pixels),np.zeros((1,1)))])
            np.testing.assert_array_equal(result,pixels)

if __name__=='__main__':unittest.main()
