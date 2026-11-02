import unittest
import numpy as np
from matching import match_descriptors

class MatchingTests(unittest.TestCase):
    def test_known_neighbors_and_ratio(self):
        r=match_descriptors(np.array([[0.]]),np.array([[4.],[1.],[8.]]))
        self.assertEqual(r['nearest_target_rows'][0],1);self.assertEqual(r['second_target_rows'][0],0)
        self.assertEqual(r['d1'][0],1);self.assertEqual(r['d2'][0],4)
        self.assertEqual(r['ratio'][0],.25);self.assertEqual(r['ratio_squared'][0],.0625)
        self.assertTrue(r['ratio_pass'][0])

    def test_strict_threshold(self):
        r=match_descriptors(np.array([[0.]]),np.array([[.7],[1.]]))
        self.assertFalse(r['ratio_pass'][0])

    def test_zero_and_duplicate_descriptors(self):
        unique=match_descriptors(np.array([[0.]]),np.array([[0.],[2.]]))
        self.assertTrue(unique['ratio_pass'][0]);self.assertEqual(unique['ratio'][0],0)
        duplicate=match_descriptors(np.array([[0.]]),np.array([[0.],[0.],[2.]]))
        self.assertFalse(duplicate['ratio_pass'][0]);self.assertTrue(np.isnan(duplicate['ratio'][0]))

    def test_validity_preserves_original_rows(self):
        r=match_descriptors(np.array([[np.nan],[0.],[2.]]),np.array([[0.],[np.nan],[8.]]),[False,True,True],[True,False,True])
        self.assertEqual(r['nearest_target_rows'][0],-1);self.assertEqual(r['nearest_target_rows'][1],0)
        self.assertEqual(r['second_target_rows'][1],2);self.assertFalse(r['ratio_pass'][0])

    def test_mutual_many_to_one(self):
        r=match_descriptors(np.array([[0.],[.2]]),np.array([[0.],[10.]]))
        np.testing.assert_array_equal(r['ratio_pass'],[True,True])
        np.testing.assert_array_equal(r['mutual_nearest'],[True,False])

    def test_equal_distance_ties_by_row(self):
        r=match_descriptors(np.array([[0.],[0.]]),np.array([[-1.],[1.]]))
        np.testing.assert_array_equal(r['nearest_target_rows'],[0,0])
        np.testing.assert_array_equal(r['ratio_pass'],[False,False])
        np.testing.assert_array_equal(r['mutual_nearest'],[True,False])

    def test_empty_insufficient_and_invalid(self):
        r=match_descriptors(np.empty((0,64)),np.ones((2,64)))
        self.assertEqual(len(r['ratio_pass']),0)
        r=match_descriptors(np.ones((2,64)),np.ones((1,64)))
        self.assertFalse(r['ratio_pass'].any());self.assertTrue((r['nearest_target_rows']==-1).all())
        with self.assertRaises(ValueError):match_descriptors(np.ones((2,3)),np.ones((2,4)))
        with self.assertRaises(ValueError):match_descriptors(np.array([[np.nan]]),np.ones((2,1)))
        with self.assertRaises(ValueError):match_descriptors(np.ones((1,1)),np.ones((2,1)),ratio_threshold=1.1)

if __name__=='__main__':unittest.main()
