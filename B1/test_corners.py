import unittest
import numpy as np
from corners import harris_response,detect_candidates,anms,working_to_original

class CornerTests(unittest.TestCase):
    def test_constant_has_no_corners(self):
        response=harris_response(np.ones((64,64)))
        np.testing.assert_allclose(response,0,atol=1e-14)
        xy,strength=detect_candidates(response,border=5)
        self.assertEqual(xy.shape,(0,2));self.assertEqual(len(strength),0)

    def test_edge_negative_and_square_corner_positive(self):
        edge=np.zeros((80,80));edge[:,40:]=1
        r=harris_response(edge);self.assertLess(r[40,40],0)
        square=np.zeros((80,80));square[25:55,25:55]=1
        response=harris_response(square);self.assertGreater(response[24:28,24:28].max(),0)
        xy,_=detect_candidates(response,border=8)
        self.assertEqual(len(xy),4)

    def test_known_radii(self):
        xy=np.array([[0,0],[3,0],[0,4],[20,0]])
        selected,r=anms(xy,np.array([100,80,60,50.]),2)
        np.testing.assert_allclose(r[1:],[3,4,17]);self.assertTrue(np.isinf(r[0]))
        np.testing.assert_array_equal(selected,[0,3])

    def test_robust_inequality_and_ties(self):
        selected,r=anms(np.array([[0,0],[3,0],[10,0]]),np.array([100,95,80.]),3)
        self.assertTrue(np.isinf(r[0]) and np.isinf(r[1]));self.assertEqual(r[2],7)
        np.testing.assert_array_equal(selected,[0,1,2])

    def test_empty_and_invalid(self):
        idx,r=anms(np.empty((0,2)),np.empty(0),500)
        self.assertEqual(len(idx),0);self.assertEqual(len(r),0)
        with self.assertRaises(ValueError):anms(np.array([[0,0]]),np.array([-1.]),1)
        with self.assertRaises(ValueError):anms(np.array([[0,0]]),np.array([1.]),1,robust=1.1)
        with self.assertRaises(ValueError):harris_response(np.full((10,10),np.nan))

    def test_original_coordinate_round_trip(self):
        p=np.array([[0,0],[1199,1599],[600,800.]])
        for size in [(4284,5712),(3024,4032)]:
            q=working_to_original(p,size,(1200,1600))
            np.testing.assert_allclose((q+.5)*np.array([1200,1600])/size-.5,p,atol=1e-12)

if __name__=='__main__':unittest.main()
