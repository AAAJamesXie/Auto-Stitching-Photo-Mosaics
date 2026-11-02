"""E2E check of saved A.2 deliverables against real photos and independent arithmetic."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


class ReportTests(unittest.TestCase):
    def test_build_exports_eight_verified_adjacent_matrices(self):
        self.assertTrue((ROOT/'build_report.py').is_file(), 'A.2 report generator is not implemented')
        result = subprocess.run([sys.executable, str(ROOT/'build_report.py')],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        results = json.loads((ROOT/'homographies.json').read_text())
        points = json.loads((ROOT/'correspondences.json').read_text())
        self.assertEqual(len(results['pairs']), 8)
        self.assertEqual(results['solved_count'], 8)
        self.assertEqual(results['unresolved_count'], 0)
        self.assertEqual([p['id'] for p in results['pairs']], ['8443_8444','8444_8445','8445_8446','8446_8447','8447_8448','8448_8450','8450_8451','8451_8452'])
        for record, pair in zip(results['pairs'], points['pairs']):
            self.assertEqual(record['id'], pair['id'])
            self.assertEqual(record['status'], 'solved')
            self.assertEqual(record['fit_count'], 12)
            self.assertEqual(record['check_count'], 4)
            for side in ['source', 'target']:
                with Image.open(ROOT.parent/'A1'/'images'/pair[side]) as photo:
                    self.assertEqual(list(photo.size), record[side+'_size'])
                coords = np.array([p[side] for p in pair['points']])
                self.assertTrue(np.all(coords >= 0))
                self.assertTrue(np.all(coords < record[side+'_size']))
            H = np.array(record['H_source_to_target'])
            self.assertEqual(H.shape, (3, 3))
            self.assertAlmostEqual(H[2, 2], 1.)
            src = np.array([p['source'] for p in pair['points']])
            dst = np.array([p['target'] for p in pair['points']])
            x, y = src.T
            den = H[2, 0]*x + H[2, 1]*y + H[2, 2]
            u = (H[0, 0]*x + H[0, 1]*y + H[0, 2])/den
            v = (H[1, 0]*x + H[1, 1]*y + H[1, 2])/den
            errors = np.sqrt((u-dst[:, 0])**2+(v-dst[:, 1])**2)
            np.testing.assert_allclose(errors, record['forward_errors_px'], atol=1e-9)
            self.assertAlmostEqual(record['fit_rmse_px'], np.sqrt(np.mean(errors[:12]**2)))
            self.assertAlmostEqual(record['check_rmse_px'], np.sqrt(np.mean(errors[12:]**2)))
            system = json.loads((ROOT/record['equations_file']).read_text())
            A = np.array(system['A_raw'])
            self.assertEqual(A.shape, (24, 8))
            a, b = src[0]; c, d = dst[0]
            np.testing.assert_allclose(A[0], [a, b, 1, 0, 0, 0, -c*a, -c*b])
            np.testing.assert_allclose(system['b_raw'][:2], [c, d])
            with Image.open(ROOT/record['figure']) as im:
                im.load()
                self.assertGreater(im.width, 1200)
                self.assertGreater(im.height, 700)
        self.assertGreater((ROOT/'report.html').stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
