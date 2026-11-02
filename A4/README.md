# A.4 - Seven views, one mosaic

Current result: report.html and results/panorama.png. Seven sources IMG_8443, 8444, 8445, 8446, 8447, 8448, 8450 share IMG_8446 reference coordinates, with a 60% destination scale. Original photos are sampled directly once by A3 bilinear inverse mapping, using composed unchanged A2 transforms and existing handwritten feathering.

Run from A4:
```powershell
python run_panorama.py
python build_report.py
python -m unittest test_panorama_geometry -v
```

panorama_measurements.json records numerical parameters, transforms, source hashes and verification samples. panorama_alignment.json records adjacent excluded-check residuals in the shared grid, not cumulative global accuracy. Quantized 16-bit source weights are presentation artifacts; actual blending uses float32 weights. results/panorama_coverage_count.png stores raw support counts 0-7.

Earlier two-image reports and numerical outputs are kept locally and excluded from this repository. The shared helper run_mosaic.py is retained because run_panorama.py imports its image utilities.

The clarification requires one Part A mosaic with >=2 images and one Part B mosaic with >=3 images. B4/report.html contains the automatic result. No new dependency. Remaining parallax, reflected-scene changes and chained-registration drift are disclosed rather than hidden.
