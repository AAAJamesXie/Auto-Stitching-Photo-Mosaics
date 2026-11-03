"""Build the A.3 local report from measured artifacts, without rerunning experiments."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent


def number(value):
    return f'{value:,}'


def case_section(case,index):
    esc=html.escape
    matrix='\n'.join('  '.join(f'{v: .9g}' for v in row) for row in case['H'])
    bench=case['benchmark']
    tag=f'R{index}' if case['kind']=='rectification' else f'P{index}'
    title=f'{tag} / {case["label"]}'
    if case['kind']=='rectification':
        # Presentation-only annotated crop; original pixels and selected quad remain explicit.
        quad=np.array(case['source_quad'])
        with Image.open(ROOT.parent/'A1'/'images'/case['image']) as im:
            x0,y0=np.maximum(0,quad.min(axis=0)-70).astype(int)
            x1,y1=np.minimum(im.size,quad.max(axis=0)+70).astype(int)
            crop=im.crop((int(x0),int(y0),int(x1),int(y1)))
            draw=ImageDraw.Draw(crop)
            pts=[tuple(p-[x0,y0]) for p in quad]
            draw.line(pts+[pts[0]],fill='#ff5a00',width=2)
            f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
            for i,(x,y) in enumerate(pts):
                draw.ellipse((x-3,y-3,x+3,y+3),fill='#ff5a00')
                draw.text((x+5,y-22),str(i+1),font=f,fill='#ff5a00',stroke_width=1,stroke_fill='black')
            crop_path='figures/'+case['id']+'_input_detail.jpg'
            crop.save(ROOT/crop_path,quality=97)
        rows=''.join(f'<tr><td>{i+1}</td><td>{p[0]}, {p[1]}</td><td>{q[0]}, {q[1]}</td></tr>' for i,(p,q) in enumerate(zip(case['source_quad'],case['target_quad'])))
        inputs=f'''<div class="two-up"><figure><a href="../A1/images/{case['image']}"><img src="{case['annotated_source']}" alt="Original photograph with selected window corners" loading="lazy"></a><figcaption>{esc(case['image'])} · Original scene and the four selected corners.</figcaption></figure><figure><a href="{crop_path}"><img src="{crop_path}" alt="Original-resolution selected rectangular frame" loading="lazy"></a><figcaption>Original-resolution crop. Corners 1–4 follow top-left, top-right, bottom-right, bottom-left.</figcaption></figure></div>
<p>{esc(case['aspect_note'].replace('Chosen rectangle uses twice the mean visible opposite-edge lengths. This is a display/aspect choice, not a recovered physical aspect ratio or metric calibration.', 'The chosen rectangle uses twice the mean visible opposite-edge lengths, not a measured physical aspect ratio.'))} The output includes 32 pixels of context.</p>
<details><summary>Selected source points and target rectangle</summary><div class="scroll"><table><thead><tr><th>Corner</th><th>Source pixel center</th><th>Target pixel center</th></tr></thead><tbody>{rows}</tbody></table></div><p>Four-corner algebraic fit residual: at most {max(case['corner_fit_errors_px']):.2g} target pixels. This checks the supplied corner mapping, not independent accuracy or physical calibration.</p></details>'''
    else:
        inputs=f'''<div class="two-up"><figure><a href="../A1/images/{case['image']}"><img src="../A1/previews/{case['image']}" alt="Original source photograph" loading="lazy"></a><figcaption>Source: {esc(case['image'])}.</figcaption></figure><figure><a href="../A1/images/{case['reference']}"><img src="../A1/previews/{case['reference']}" alt="Reference photograph whose coordinate frame is used" loading="lazy"></a><figcaption>Reference coordinate frame: {esc(case['reference'])}.</figcaption></figure></div>
<p>A.2 H warps the full source; four transformed corners define its canvas. The reference is shown for context, not blended here. The A.2 excluded-check RMSE was {case['A2_check_rmse_px']:.3f} target pixels, measured on local architectural points.</p>'''
    x,y,w,h=case['canvas']
    valid=case['full_resolution']['nearest']['valid_pixels']
    links=' · '.join(f'<a href="{case["outputs"][m]}">{label} full-resolution RGBA PNG</a> · <a href="{case["masks"][m]}">{label} validity mask</a>' for m,label in [('nearest','Nearest'),('bilinear','Bilinear')])
    return f'''<section class="report-section" id="{case['id']}"><p class="eyebrow">{tag} · {case['kind'].upper()}</p><h2>{esc(title)}</h2>{inputs}
<figure class="result-figure"><a href="{case['overview']}"><img src="{case['overview']}" alt="Nearest-neighbor and bilinear outputs of {esc(case['label'])}" loading="lazy"></a><figcaption>Same H, output origin ({x}, {y}), and {number(w)} × {number(h)} destination grid. {number(valid)} pixel centers are valid under both methods. Gray outside the footprint represents transparent pixels.</figcaption></figure>
<figure class="result-figure"><a href="{case['detail']}"><img src="{case['detail']}" alt="Identical output crop comparing interpolation pixel structure" loading="lazy"></a><figcaption>Identical canvas crop {case['detail_canvas_box']}, with each saved pixel replicated 4× for both methods. Nearest looks stepped; bilinear is smoother. Neither corrects geometry, reflections or parallax.</figcaption></figure>
<p class="resource-links">{links}</p>
<details><summary>Recovered H and canvas metadata</summary><pre>{esc(matrix)}</pre><p>World origin: ({x}, {y}); output array: {w} × {h}. Subtract the origin for canvas indices. Full transforms, masks and timing data are linked.</p></details>
</section>'''


def main():
    data=json.loads((ROOT/'measurements.json').read_text(encoding='utf-8'))
    rects=[c for c in data['cases'] if c['kind']=='rectification']
    projections=[c for c in data['cases'] if c['kind']=='projection']
    if len(rects)!=2 or len(projections)!=2:
        raise ValueError('The report requires exactly two rectifications and two adjacent projections.')
    sections=''.join(case_section(c,i+1) for group in [rects,projections] for i,c in enumerate(group))
    timing_rows=[]
    for i,c in enumerate(rects+projections):
        tag=('R'+str(i+1)) if i<2 else ('P'+str(i-1))
        b=c['benchmark'];n=b['median_seconds']['nearest'];bil=b['median_seconds']['bilinear']
        timing_rows.append(f'<tr><td>{tag}</td><td>{number(b["destination_pixels"])}</td><td>{n*1000:.2f}</td><td>{bil*1000:.2f}</td><td>{bil/n:.2f}×</td></tr>')
    ratios=[c['benchmark']['bilinear_to_nearest_time_ratio'] for c in data['cases']]
    table=''.join(timing_rows)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A.3 · Warping &amp; Rectification</title><link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="page.css"></head><body class="report-page"><main class="shell">
<header class="site-header"><a class="brand" href="../index.html">Stitching Photo<br>Mosaics</a><nav class="header-nav" aria-label="A.3 navigation"><a href="../index.html">Project home <span>01</span></a><a href="#rectifications">Rectifications <span>02</span></a><a href="#timing">Speed &amp; quality <span>03</span></a></nav></header>
<section class="project-hero"><p class="eyebrow">SYDE 671 · Assignment 3 · James Xie</p><h1>A.3 / WARPING<br>&amp; RECTIFICATION</h1><p class="project-deck">Inverse-map the output grid. Sample the source by hand. Compare two ways to reconstruct the image.</p><div class="project-hero-rule"><span>Part A · Warp the images</span><span>2 rectifications · 2 adjacent projections · 2 interpolators</span></div></section>
<section class="report-section" id="method"><h2>01 / Every output pixel finds its source.</h2><p>For each output pixel (x, y), apply H⁻¹ and divide by the homogeneous coordinate to find source position (s, t). Inverse mapping avoids gaps from forward scattering.</p><div class="formula">[a, b, c]ᵀ = inverse(H) · [x, y, 1]ᵀ
s = a / c       t = b / c</div><div class="two-up"><div><h3>Nearest neighbor</h3><p>Sample floor(s + 0.5), floor(t + 0.5), rounding half ties upward. Values stay discrete, but diagonal edges can look stepped.</p></div><div><h3>Bilinear</h3><p>Let i = floor(s), j = floor(t), α = s − i, and β = t − j. Blend the four neighboring source samples using their two-dimensional distance weights.</p></div></div><div class="formula">I(s,t) = (1−α)(1−β) I[j,i] + α(1−β) I[j,i+1]
       + (1−α)β I[j+1,i] + αβ I[j+1,i+1]</div><p>Both samplers are handwritten in NumPy. Linear algebra handles H; Pillow handles image I/O and figures. No high-level warp or interpolation API is used.</p></section>
<section class="report-section"><h2>02 / Bounds, origins &amp; transparent pixels.</h2><p>Integer coordinates denote pixel centers. Bounds come from the four transformed corners, using floor/ceil and inclusive endpoints. The origin preserves negative coordinates; row blocks limit memory.</p><p>Automatic bounds reject a projective horizon crossing the source. A finite destination ROI can still be sampled. Outputs above 40 million pixels are rejected before allocation.</p><p>Both masks use 0 ≤ s ≤ width−1 and 0 ≤ t ≤ height−1. Invalid samples are transparent. Bilinear sampling clamps neighbors at exact borders. Valid black pixels stay opaque.</p><p>The functions return a <code>WarpResult</code> containing <code>image</code> (float32), <code>alpha</code> (boolean), <code>origin</code> and <code>H_source_to_canvas</code>. Saved PNGs round RGB values to uint8 and retain an explicit alpha channel.</p></section>
<section class="report-section" id="rectifications"><h2>03 / Two real rectangular windows.</h2><p>Two windows are rectified by mapping their four frame corners to axis-aligned rectangles with the A.2 solver. Only the selected plane is straightened.</p><div class="note">The physical window aspect ratios are not known. Output aspect ratios are chosen from the visible opposite-edge lengths, at roughly 2× their observed scale. These are documented presentation choices, not metric measurements of the building.</div></section>
''' + sections + f'''
<section class="report-section" id="timing"><h2>04 / Speed versus sample quality.</h2><p>Each row compares identical inputs, H, ROI, masks and block size. After warm-up, report the median of three runs. Timing includes validation, allocation and sampling, but excludes I/O and figures.</p><div class="scroll"><table><thead><tr><th>Case</th><th>ROI pixels</th><th>Nearest (ms)</th><th>Bilinear (ms)</th><th>Time ratio</th></tr></thead><tbody>{table}</tbody></table></div><p>The ROI is centered in the full output and bounded to 512 × 512 pixels. R1 and R2 are the front and side windows; P1 and P2 are the front and side adjacent-view projections. Compare methods within each row; image sizes differ. Timings reflect this machine and three runs.</p><p>Here bilinear takes {min(ratios):.2f}–{max(ratios):.2f}× the nearest-neighbor time. Four weighted samples give smoother edges but can soften detail. Neither method properly filters strong minification.</p><p>The saved mean absolute output difference measures sample disagreement, not reconstruction accuracy; no aligned ground truth is available.</p></section>
<section class="report-section"><h2>05 / What the examples establish.</h2><p>The examples verify rectification, full-image bounds and transparent support. Different depths, reflections and vegetation may still disagree. A.4 shows the blended panorama.</p><p>Sampling uses stored RGB values, without radiance linearization. Overviews use nearest display scaling; details use equal integer pixel replication. Full RGBA outputs are linked.</p></section>
<section class="report-section"><h2>Reproduce &amp; inspect.</h2><pre>python run_experiments.py
python build_report.py
python -m unittest test_warp -v</pre><p><a href="warp.py">Handwritten inverse warp and both samplers</a> · <a href="rectification_points.json">Selected corners and target rectangles</a> · <a href="measurements.json">All matrices, bounds and timing samples</a> · <a href="run_experiments.py">Experiment source</a> · <a href="test_warp.py">Unit tests</a> · <a href="../A2/report.html">A.2 homography report</a></p><p>Checks cover known samples and transforms, valid black pixels, borders, masks, bounds and invalid inputs. Real-photo outputs and input hashes were independently verified.</p><p class="method-note">Python {data['software']['python']} · NumPy {data['software']['numpy']} · Pillow {data['software']['pillow']}. No new dependency was installed.</p></section>
<nav class="project-home-nav" aria-label="Project navigation"><a class="solid-link" href="../index.html"><span aria-hidden="true">&larr;</span>Back to Project Home</a></nav>
<footer class="site-footer"><span>SYDE 671 · Assignment 3 · James Xie</span><span>A.3 · Warping &amp; rectification</span></footer></main></body></html>'''
    (ROOT/'report.html').write_text(page,encoding='utf-8')
    (ROOT/'README.md').write_text('''# A.3 - Warping and Rectification

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
''',encoding='utf-8')
    print('A.3 report and README saved from four measured experiments.')


if __name__=='__main__':
    main()
