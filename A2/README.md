# A.2 - Recovering Homographies

Open `report.html` for the equations, numbered correspondence figures, recovered matrices, measured errors, and limitations for all eight adjacent pairs.

## Files

- `homography.py`: handwritten `computeH(im1_pts, im2_pts)` using normalized eight-unknown linear least squares; H maps image 1 into image 2.
- `manual_seeds.json`: visually selected architectural seeds; annotation revisions are disclosed per point.
- `prepare_correspondences.py`: optional bounded local NCC refinement of each manually seeded target. This is assisted annotation, not unassisted mouse-click annotation or the Part B feature-matching pipeline.
- `correspondences.json`: 16 coordinate pairs per solved adjacent pair, 12 used for fitting and four excluded checks.
- `homographies.json`: original-pixel homographies and forward reprojection errors.
- `equations/`: full raw and normalized linear systems with normalization transforms; raw CSV exports are also supplied.
- `figures/`: numbered correspondence crops and all local source/target patch pairs.
- `inspection/`: coordinate grids, detailed crops, and initial annotation snapshots used during QA.
- `test_homography.py`, `test_refinement.py`, `test_report.py`: targeted unit and actual-output e2e tests, without mocks.

## Reproduce

```powershell
python prepare_correspondences.py
python build_report.py
python -m unittest test_homography test_refinement test_mixed_resolution test_report -v
```

Requires the already installed NumPy, Pillow and OpenCV. No dependency was installed by this task.

Coordinates are full-resolution pixel centers, x right, y down. Every supplied fit point participates equally; there is no RANSAC or model-error-based rejection. The four check points are excluded from fitting but were visually corrected during annotation QA; they are not a blind validation set.

No high-level homography or perspective-warp functions are used. No image has been stitched, projectively warped, or resampled by H. The imported HEIC originals remain unchanged; A1 selections and previews were updated for this revision. The full-image panorama quality and fixed-camera-center assumption remain unverified until later stages.

## Measurements

- IMG_8443.jpg -> IMG_8444.jpg: fit RMSE 0.339 px; excluded-check RMSE 1.236 px; maximum check error 2.305 px.
- IMG_8444.jpg -> IMG_8445.jpg: fit RMSE 0.897 px; excluded-check RMSE 2.091 px; maximum check error 2.771 px.
- IMG_8445.jpg -> IMG_8446.jpg: fit RMSE 0.288 px; excluded-check RMSE 0.687 px; maximum check error 1.133 px.
- IMG_8446.jpg -> IMG_8447.jpg: fit RMSE 0.567 px; excluded-check RMSE 1.611 px; maximum check error 2.415 px.
- IMG_8447.jpg -> IMG_8448.jpg: fit RMSE 0.869 px; excluded-check RMSE 0.838 px; maximum check error 1.376 px.
- IMG_8448.jpg -> IMG_8450.jpg: fit RMSE 1.067 px; excluded-check RMSE 2.289 px; maximum check error 2.777 px.
- IMG_8450.jpg -> IMG_8451.jpg: fit RMSE 1.091 px; excluded-check RMSE 1.971 px; maximum check error 2.310 px.
- IMG_8451.jpg -> IMG_8452.jpg: fit RMSE 0.811 px; excluded-check RMSE 1.924 px; maximum check error 2.558 px.
