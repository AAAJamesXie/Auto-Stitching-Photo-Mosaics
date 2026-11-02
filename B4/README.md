# B.4 - One automatic seven-image mosaic

Open report.html or results/automatic_mosaic.png. Seven photos8443/8444/8445/8446/8447/8448/8450, reference8446,60% output grid. Exactly one new automatic mosaic; results/manual_mosaic.png is the A4 same-input baseline on the identical canvas.

ransac.py implements four-point RANSAC from B3 ratio<.7 plus mutual matches. Fixed symmetric RMS threshold3 working pixels, seeds1337-1342, minimum100 tested hypotheses/max5000 draws, nominal confidence.999, >=8 inliers, iterative least-squares refit with20-round cap. All6 actual models converge;468/501 input pairs support their final models. This is not an independent accuracy percentage.

run_ransac.py freezes automatic H before manual matrices are consulted. run_mosaic.py then compares original-coordinate transforms, handwritten inverse bilinear sampling and feathering on a common6623x3165 grid. Source images are not pre-resized. The manual baseline exactly equals prior A4 PNG pixels at its saved origin. Same image support, alpha and45 scalar samples per method are checked.

Run from B4:
```powershell
python run_ransac.py
python run_mosaic.py
python build_report.py
python -m unittest test_ransac -v
```

data/ holds all point identities, input masks, best four-point sample, initial/final/refit masks, working and original H, symmetric errors and iteration histories. automatic_homographies.json and mosaic_measurements.json record provenance/hashes. No built-in RANSAC/homography/warp/stitching API and no new dependency. Seven analytical unit tests passed. Reflections, parallax and accumulated drift remain disclosed limitations.
