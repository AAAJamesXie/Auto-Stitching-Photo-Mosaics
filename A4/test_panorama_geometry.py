import unittest
import numpy as np
from panorama_geometry import compose_transforms

class GeometryTests(unittest.TestCase):
    def test_chain_order_and_inverse(self):
        names=['a','b','c'];a=np.array([[1,0,-10],[0,1,2],[0,0,1.]])
        b=np.array([[1,0,-20],[0,1,3],[0,0,1.]])
        t=compose_transforms(names,1,{'a_b':a,'b_c':b},1)
        np.testing.assert_allclose(t[0],a);np.testing.assert_allclose(t[1],np.eye(3));np.testing.assert_allclose(t[2],np.linalg.inv(b))

    def test_noncommuting_composition(self):
        a=np.array([[2,0,5],[0,1,0],[0,0,1.]])
        b=np.array([[1,0,3],[0,2,7],[0,0,1.]])
        t=compose_transforms(['a','b','c'],2,{'a_b':a,'b_c':b},1)
        np.testing.assert_allclose(t[0],b@a)

    def test_pixel_center_scale(self):
        t=compose_transforms(['a'],0,{},.6)[0]
        np.testing.assert_allclose(t@np.array([0,0,1]),[-.2,-.2,1])
        np.testing.assert_allclose(t@np.array([10,20,1]),[5.8,11.8,1])

if __name__=='__main__':unittest.main()
