# B.2 - Axis-aligned patch descriptors

Open report.html for the method, seven datasets and six spatial examples per photograph. B1 working PNGs and ANMS row identity are preserved. No descriptors are matched in this stage.

descriptors.py uses full-image Gaussian sigma2/kernel13/REFLECT_101, symmetric40x40 bilinear sample-center offsets-19.5..19.5, region indices2,7,...,37 to obtain8x8 (world offsets-17.5,-12.5,...17.5), then row-major flatten and population mean/std normalization. std must exceed1e-6; invalid rows retain zero vectors and false validity. Numerical values are never clipped. No rotation/scale normalization or Haar transform.

data/*.npz stores descriptors, sampled8, mean/std, flags, B1 rows/candidate indices, feature IDs, working/original coordinates and transforms. data/*.json stores row metadata/status. examples/*.npz preserves raw40/blur40/8x8/normalized arrays for six spatial examples. Figures use fixed display scales; normalized display clips to[-3,3] without clipping numerical data.

Run from B2:
```powershell
python run_descriptors.py
python build_report.py
python -m unittest test_descriptors -v
```

Six analytical unit checks passed. The current seven datasets each contain500 valid64D vectors. No new dependencies; upstream inputs are hash-checked unchanged. B3 matching and B4 automatic mosaicing remain upcoming.
