import unittest
import numpy as np
from ransac import symmetric_errors,degenerate_sample,ransac_homography,working_h_to_original

def project(p,H):
    q=np.c_[p,np.ones(len(p))]@H.T
    return q[:,:2]/q[:,2,None]

class RansacTests(unittest.TestCase):
    def test_symmetric_error_and_homogeneous_scale(self):
        p=np.array([[10,20],[100,50],[200,300],[400,100.]])
        H=np.array([[1.1,.03,20],[-.02,.98,-10],[.0001,-.0002,1.]])
        q=project(p,H)
        np.testing.assert_allclose(symmetric_errors(p,q,H),0,atol=1e-10)
        np.testing.assert_allclose(symmetric_errors(p,q,H*-1e-20),0,atol=1e-10)
        q=q+[2,0];self.assertTrue((symmetric_errors(p,q,H)>1).all())

    def test_degenerate_triplets_and_duplicates(self):
        self.assertTrue(degenerate_sample(np.array([[0,0],[1,0],[2,0],[0,1.]])))
        self.assertTrue(degenerate_sample(np.array([[0,0],[0,0],[1,1],[0,1.]])))
        self.assertFalse(degenerate_sample(np.array([[0,0],[1,0],[1,1],[0,1.]])))

    def test_known_noisy_model_with_outliers(self):
        rng=np.random.default_rng(6);p=rng.uniform(50,900,(60,2))
        H=np.array([[1.03,.015,40],[-.01,.99,-20],[.00004,-.00002,1.]])
        q=project(p,H)+rng.normal(0,.08,(60,2));q[40:]=rng.uniform(0,1000,(20,2))
        r=ransac_homography(p,q,seed=42,max_iterations=600,min_iterations=60,threshold=1.)
        self.assertGreaterEqual(r['inliers'][:40].sum(),39);self.assertFalse(r['inliers'][40:].any())
        self.assertLess(np.median(np.linalg.norm(project(p[:40],r['H'])-project(p[:40],H),axis=1)),.1)

    def test_seed_reproducible(self):
        rng=np.random.default_rng(8);p=rng.uniform(0,500,(20,2));q=p+[20,-10];q[-3:]=rng.uniform(0,500,(3,2))
        a=ransac_homography(p,q,seed=9,max_iterations=100,min_iterations=20)
        b=ransac_homography(p,q,seed=9,max_iterations=100,min_iterations=20)
        np.testing.assert_array_equal(a['inliers'],b['inliers']);np.testing.assert_array_equal(a['H'],b['H'])
        self.assertEqual(a['draws'],b['draws'])

    def test_no_consensus(self):
        rng=np.random.default_rng(4);p=rng.uniform(0,1000,(15,2));q=rng.uniform(0,1000,(15,2))
        with self.assertRaises(ValueError):ransac_homography(p,q,threshold=.001,max_iterations=80,min_iterations=20)

    def test_original_coordinate_conversion(self):
        a=np.array([[.28,0,-.36],[0,.28,-.36],[0,0,1.]])
        b=np.array([[.4,0,-.3],[0,.4,-.3],[0,0,1.]])
        H=np.array([[1,0,100],[0,1,-20],[0,0,1.]])
        orig=working_h_to_original(H,a,b);p=np.array([[100,200],[300,400.]])
        np.testing.assert_allclose(project(project(p,a),H),project(project(p,orig),b),atol=1e-10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):ransac_homography(np.zeros((3,2)),np.zeros((3,2)))
        with self.assertRaises(ValueError):ransac_homography(np.full((10,2),np.nan),np.ones((10,2)))
        with self.assertRaises(ValueError):ransac_homography(np.zeros((10,2)),np.ones((10,2)),threshold=-1)

if __name__=='__main__':unittest.main()
