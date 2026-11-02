# [Auto] Stitching Photo Mosaics

SYDE 671 Assignment 3 · James Xie

Manual and automatic stitching of architectural photographs. The current Part A and Part B mosaics each use seven images, meeting the clarified requirements of one >=2-image Part A mosaic and one >=3-image Part B mosaic.

## View the reports

Live site: [Stitching Photo Mosaics](https://aaajamesxie.github.io/Auto-Stitching-Photo-Mosaics/).

GitHub Pages publishes the root of the main branch. The .nojekyll file serves the existing static files directly; updates are published after pushing to main. For offline viewing, clone or download the repository and open [index.html](index.html). Images, reports and data links use relative paths.

- [A.1 photographs and notes](A1/README.md): nine selected views from one acquisition set. The original two-set A.1 request is not fully covered by this single set.
- [A.2 homographies](A2/report.html): handwritten normalized least squares, annotated points and error diagnostics.
- [A.3 warping](A3/report.html): handwritten nearest-neighbor/bilinear sampling and two rectifications.
- [A.4 manual mosaic](A4/report.html): seven images with feathered blending.
- [B.1 corners](B1/report.html): Harris response and handwritten ANMS.
- [B.2 descriptors](B2/report.html): axis-aligned40x40 support reduced to normalized8x8 patches.
- [B.3 matching](B3/report.html): nearest/second-nearest search, ratio test and mutual flags.
- [B.4 automatic mosaic](B4/report.html): handwritten four-point RANSAC, seven-image stitching and manual comparison.

## Source and reproduction

Each report links its numerical records and implementation. Report generators rebuild the HTML; the stage experiment scripts regenerate numerical data and figures. Use the existing Python environment with NumPy, Pillow and OpenCV. Rendering scripts use Windows Arial fonts. Focused numerical test sources are retained because reports and reproduction commands reference them.

The HTML explanations were shortened without changing formulas, parameters or numerical outputs. PNG/JPG files and linked NPZ/JSON/CSV data are stored directly in Git. Exact file bytes are preserved so recorded input hashes remain meaningful.

## Repository scope

The allowlist in [.gitignore](.gitignore) tracks the current reports, required images/results, core scripts and small linked numerical data. HEIC originals, historical reports, backups, inspection screenshots, browser QA, one-off editing tools, caches and plan files remain local. The older two-image A.4 report is excluded; its shared image-helper source is retained for current-script imports.

Limitations are documented in the reports: reflections, multiple depths and chained alignment can leave local mismatch; fitting residuals do not measure independent full-scene accuracy.
