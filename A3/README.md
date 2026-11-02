# A.3 - Warping and Rectification

Open report.html for the two real-window rectifications, two adjacent-photo projections, both interpolation methods, original-pixel corner annotations, masks, timing data and limitations.

- warp.py implements warpImageNearestNeighbor(im,H) and warpImageBilinear(im,H) from scratch with NumPy. They return WarpResult(image,alpha,origin,H_source_to_canvas). Both accept optional canvas=(world_x0,world_y0,width,height), block_rows=64 and max_pixels=40000000.
- rectification_points.json records full-resolution source frame corners and chosen target rectangles. Physical aspect ratios are not known; displayed ratios are based on visible edge lengths.
- run_experiments.py reuses the handwritten A2 solver/matrices, saves full-resolution RGBA PNGs and masks, compares identical output crops and benchmarks the same ROI per method.
- build_report.py regenerates the local report from measurements.json without rerunning the experiments.
- results/ holds numerical warp outputs; figures/ holds annotation and presentation derivatives; inspection/ holds visual QA.
- input_hashes.json records all A1 JPG and A2 solver/matrix input hashes, checked unchanged after experiment generation.

Run from this A3 folder:

```powershell
python run_experiments.py
python build_report.py
python -m unittest test_warp -v
```

No high-level image-warp or interpolation library computes an experiment. Pillow resizing is only for presentation thumbnails. The final RGB PNG export rounds float32 results to uint8; alpha and standalone validity masks retain the valid sampling domain. Adjacent projections are resampled source images, without blending or mosaicing.
