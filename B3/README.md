# B.3 - Descriptor matching

Open report.html. Six adjacent pairs from8443,8444,8445,8446,8447,8448,8450. Current primary Euclidean d1/d2<0.7 gives552 tentative matches;501 additionally satisfy mutual-nearest consistency. These are not geometric inliers or a measured correctness rate.

matching.py performs exact float64 direct squared-difference search with64-query blocks and stable target-row ties. Invalid descriptors are excluded; second neighbor is a distinct target row. Zero-second-distance0/0 rejects; fewer than2 valid targets yields no ratio match. Euclidean ratio and squared-distance ratio are both saved;0.7 Euclidean is mathematically0.49 squared. Reverse-nearest flags use all valid sources, without reverse ratio filtering.

data/*.npz stores every source-query audit array and all ratio-pass row pairs/feature IDs/candidate indices/working and original coordinates. Match arrays follow ascending source row; query mutual flags join by match_source_rows. data/*.json contains all500 query metadata records per pair. Figure overviews show up to60 ranks spread across the ratio-ordered accepted set; full accepted-line images are separate. Local examples use lowest/median/highest accepted ratios and lowest rejected ratio, without geometry labels.

Run from B3:
```powershell
python run_matching.py
python build_report.py
python -m unittest test_matching -v
```

Seven analytical unit tests passed. Actual generation checks20 direct scalar neighbor lists per pair, squared/Euclidean ratio consistency and unchanged upstream hashes. No A2 H guides matching; no RANSAC, automatic mosaic, new dependency or publishing. B4 remains upcoming.
