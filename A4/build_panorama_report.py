"""Generate seven-image A4 report from saved numerical artifacts."""
import html,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent

def project(points,H):
    q=np.c_[points,np.ones(len(points))]@H.T
    return q[:,:2]/q[:,2,None]

def main():
    data=json.loads((ROOT/'panorama_measurements.json').read_text(encoding='utf-8'))
    layers=data['layers'];lookup={x['image']:np.array(x['H_original_to_canvas']) for x in layers}
    correspondences=json.loads((ROOT.parent/'A2/correspondences.json').read_text(encoding='utf-8'))
    diagnostics=[]
    for pair in correspondences['pairs']:
        if pair['source'] not in lookup or pair['target'] not in lookup:continue
        pts=[p for p in pair['points'] if p['role']=='check']
        a=project(np.array([p['source'] for p in pts]),lookup[pair['source']]);b=project(np.array([p['target'] for p in pts]),lookup[pair['target']])
        errors=np.linalg.norm(a-b,axis=1)
        diagnostics.append({'pair':pair['id'],'count':len(pts),'canvas_errors_px':errors.tolist(),'rmse_canvas_px':float(np.sqrt(np.mean(errors**2))),'max_canvas_px':float(errors.max())})
    (ROOT/'panorama_alignment.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    with Image.open(ROOT/'figures/panorama.jpg') as im:foot=im.convert('RGB')
    sx=foot.width/data['canvas'][2];sy=foot.height/data['canvas'][3];draw=ImageDraw.Draw(foot)
    colors=['#ff3311','#ff9900','#c6cd00','#00d06a','#00b8e6','#405cff','#ed45d7']
    for layer,color in zip(layers,colors):
        w,h=layer['original_size'];pts=project(np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]),np.array(layer['H_original_to_canvas']))
        pts=[(float(x*sx),float(y*sy)) for x,y in pts];draw.line(pts+[pts[0]],fill=color,width=3)
    foot.save(ROOT/'figures/panorama_footprints.jpg',quality=96)
    sources=''.join(f'<figure><a href="../A1/images/{x["image"]}"><img src="../A1/previews/{x["image"]}" alt="Source photograph {x["image"]}" loading="lazy"></a><figcaption>{i+1:02d} / {x["image"]}</figcaption></figure>' for i,x in enumerate(layers))
    matrices=''.join('<details><summary>'+html.escape(x['image'])+' → shared canvas</summary><pre>'+html.escape('\n'.join('  '.join(f'{v: .9g}' for v in row) for row in x['H_original_to_canvas']))+'</pre></details>' for x in layers)
    diagnostic_rows=''.join(f'<tr><td>{d["pair"].replace("_"," → ")}</td><td>{d["rmse_canvas_px"]:.3f}</td><td>{d["max_canvas_px"]:.3f}</td></tr>' for d in diagnostics)
    mask_links=' · '.join(f'<a href="results/panorama_{x["image"][4:8]}_mask.png">{x["image"][4:8]} mask</a>' for x in layers)
    w,h=data['canvas'][2:];origin=data['canvas'][:2]
    details=''.join(f'<figure class="result-figure"><a href="figures/panorama_detail_{i+1}.jpg"><img src="figures/panorama_detail_{i+1}.jpg" alt="Identical crop comparing overwrite and feathering" loading="lazy"></a><figcaption>Detail {i+1} / Canvas crop {box}. Same 240 × 240 output region, pixels replicated 3×. Left: overwrite. Right: feathering. Feathering cannot correct misalignment.</figcaption></figure>' for i,box in enumerate(data['detail_canvas_boxes']))
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A.4 · Seven Views, One Mosaic</title><link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="../A3/page.css"><link rel="stylesheet" href="page.css"></head>
<body class="report-page"><main class="shell">
<header class="site-header"><a class="brand" href="../index.html">Stitching Photo<br>Mosaics</a><nav class="header-nav" aria-label="A.4 navigation"><a href="../index.html">Project home <span>01</span></a><a href="#result">The panorama <span>02</span></a><a href="#method">Seven contributions <span>03</span></a></nav></header>
<section class="project-hero"><p class="eyebrow">SYDE 671 · Assignment 3 · James Xie</p><h1>A.4 / SEVEN VIEWS.<br>ONE MOSAIC.</h1><p class="project-deck">Follow the gold facade from its left edge to the building’s right-hand turn. Seven overlapping views become one wider image.</p><div class="project-hero-rule"><span>Part A · Image mosaicing</span><span>7 photos · 1 mosaic · 60% destination scale</span></div></section>
<section class="report-section" id="result"><h2>01 / A wider view of the building.</h2><figure class="result-figure"><a href="results/panorama.png"><img src="figures/panorama.jpg" alt="Seven-image panorama spanning the reflective building facade from left to right" loading="eager"></a><figcaption>IMG_8443 → 8444 → 8445 → 8446 → 8447 → 8448 → 8450. Final {w:,} × {h:,} RGBA canvas. Gray marks transparent pixels; all valid source support is retained.</figcaption></figure><p><a class="solid-link" href="results/panorama.png">Open the seven-image mosaic <span>→</span></a></p><p>IMG_8446 is the reference. Seven views cover the left facade through the right-hand turn, with IMG_8448 bridging IMG_8447 and IMG_8450.</p><p class="method-note">Clarification: Part A requires one mosaic with at least two images; this result uses seven. Part B requires one mosaic with at least three, shown in B.4.</p></section>
<section class="report-section"><h2>02 / The seven source images.</h2><div class="source-strip">{sources}</div><p>Three originals are 4284 × 5712; four are 3024 × 4032. The A.3 bilinear sampler reads originals directly into a 60% reference grid, without pre-resizing.</p></section>
<section class="report-section" id="method"><h2>03 / One coordinate frame, seven contributions.</h2><p>A.2 H values map each image to the next. Chain them forward on the left of IMG_8446 and inversely on the right, then apply pixel-center scaling to every image.</p><div class="formula">p_scaled = S · H_image→8446 · p_original
S = [[0.6, 0, −0.2], [0, 0.6, −0.2], [0, 0, 1]]
p_canvas = translate(−origin_x, −origin_y) · p_scaled</div><p>The four transformed corner centers of each source define its bounds; their union gives origin ({origin[0]}, {origin[1]}) and the common {w:,} × {h:,} grid. The full-scale union is about 58 million pixels; this output has {w*h:,}. RGB and weights are warped in blocks and blended sequentially.</p><figure class="result-figure"><a href="figures/panorama_footprints.jpg"><img src="figures/panorama_footprints.jpg" alt="Seven transformed source-image corner footprints over the final mosaic" loading="lazy"></a><figcaption>Transformed image boundaries, in source order: red, orange, yellow-green, green, cyan, blue and magenta. The shared grid retains each image’s full valid support.</figcaption></figure>
<p>Feather weights use distance to the source boundary, normalized to one at the center. Borders keep positive weight. Weights share the RGB warp and are zero outside valid support.</p><div class="formula">wᵢ = warp(min(x+1, width−x, y+1, height−y) / max_distance)
C = Σᵢ wᵢ Cᵢ / Σᵢ wᵢ     alpha = (Σᵢ wᵢ > 0)</div><p>Only valid samples contribute. Single-source pixels keep their values; absent pixels stay transparent. Blending uses encoded RGB, without exposure compensation or radiance calibration.</p><figure class="result-figure"><a href="figures/panorama_contributions.jpg"><img src="figures/panorama_contributions.jpg" alt="Seven normalized contribution maps shown on an identical shared canvas" loading="lazy"></a><figcaption>Darker means greater contribution on the shared frame. Display weights are 16-bit; blending uses float32.</figcaption></figure><figure class="result-figure"><a href="figures/panorama_coverage.jpg"><img src="figures/panorama_coverage.jpg" alt="Number of valid input images covering each output location" loading="lazy"></a><figcaption>Coverage count: gray = 0, red = 1, gold = 2, green = 3, teal = 4, blue = 5, purple = 6, dark = 7. {data['valid_pixels']:,} valid centers; {data['overlap_pixels']:,} have at least two inputs. <a href="results/panorama_coverage_count.png">Raw count PNG</a> stores integer counts 0–7.</figcaption></figure></section>
<section class="report-section"><h2>04 / Overwrite versus feathering.</h2><figure class="result-figure"><a href="figures/panorama_comparison.jpg"><img src="figures/panorama_comparison.jpg" alt="Stacked seven-image mosaic comparison: direct overwrite and normalized feathering" loading="lazy"></a><figcaption>Same inputs and canvas. Top: direct overwrite. Bottom: feathering. Both show the same mosaic.</figcaption></figure>{details}<p>Feathering reduces seams but leaves doubled or soft edges where frames, reflections and other planes disagree.</p></section>
<section class="report-section"><h2>05 / Check the geometry without hiding its limits.</h2><p>The table measures four excluded A.2 check pairs per neighbor on the 60% canvas. Annotations were visually revised, so this is not blind validation. Local residuals do not measure cumulative drift.</p><div class="scroll"><table><thead><tr><th>Adjacent pair</th><th>Check RMSE (canvas px)</th><th>Maximum (canvas px)</th></tr></thead><tbody>{diagnostic_rows}</tbody></table></div><p>There is no independent end-view closure check. Multiple depths, reflections, vegetation and camera translation can still disagree. Full-size output and crops show the remaining mismatch.</p><details><summary>All seven original-to-canvas matrices</summary>{matrices}</details></section>
<section class="report-section"><h2>Reproduce &amp; inspect.</h2><pre>python run_panorama.py
python build_report.py
python -m unittest test_panorama_geometry -v</pre><p><a href="run_panorama.py">Seven-image experiment</a> · <a href="panorama_geometry.py">Transform composition</a> · <a href="blend.py">Handwritten feathering and blending</a> · <a href="../A3/warp.py">Handwritten bilinear sampler</a> · <a href="panorama_measurements.json">Matrices, input hashes, support counts and 45 scalar probes</a> · <a href="panorama_alignment.json">Adjacent check geometry</a></p><p><a href="results/panorama_overwrite.png">Overwrite baseline RGBA</a> · <a href="results/panorama_union_mask.png">Union validity</a> · {mask_links}</p><p>Checks verify transform order, pixel-center scaling, masks, single-source pixels and 45 blend samples. Input hashes are unchanged.</p></section>
<nav class="project-home-nav" aria-label="Project navigation"><a class="solid-link" href="../index.html"><span aria-hidden="true">&larr;</span>Back to Project Home</a></nav>
<footer class="site-footer"><span>SYDE 671 · Assignment 3 · James Xie</span><span>A.4 · Seven views, one mosaic</span></footer></main></body></html>'''
    (ROOT/'report.html').write_text(page,encoding='utf-8')
    (ROOT/'README.md').write_text('''# A.4 - Seven views, one mosaic

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
''',encoding='utf-8')
    print('Seven-image report generated; adjacent canvas RMSE:',[round(d['rmse_canvas_px'],3) for d in diagnostics])

if __name__=='__main__':main()
