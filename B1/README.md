# B.1 - Harris corners and ANMS

Open report.html for all7 images, each with full candidate, strongest500 and ANMS500 overlays. Original input photos and Part A numerical artifacts are unchanged.

corners.py computes the classical single-scale Harris response, positive3x3 local maxima and exact blocked ANMS with robust0.9. Working images are1200x1600 lossless PNGs, resized from original RGB by Lanczos. Response threshold0.001*max, border24, derivative sigma1, integration sigma1.5, k0.04. No built-in corner detector or matching API.

points/*.npz stores every candidate's (x,y) working/original coordinates, response, suppression radius (inf when no sufficiently stronger neighbor), both500-point selections and coordinate transform. points/*.json stores selected ANMS points, with null+radius_unbounded for infinite radii. working/*.png preserves exact source samples for B2.

Run from B1:
```powershell
python run_corners.py
python build_report.py
python -m unittest test_corners -v
```

Six analytical unit checks passed. Real artifacts verify20 sampled radii per image, selected-index validity, coordinate transforms and unchanged input hashes. Spatial metrics are descriptive, not matching or mosaic accuracy. Single scale and integer feature centers only; no descriptors, matching, RANSAC or automatic mosaic. No new dependencies.

ANMS source: https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf section3. The paper uses a harmonic-mean corner response; this implementation uses classical Harris det-ktrace², an explicitly documented choice.
