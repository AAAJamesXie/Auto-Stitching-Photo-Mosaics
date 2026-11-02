"""Generate A.2 matrices, equation data, correspondence figures, and HTML report."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from homography import computeH, project_points, build_system, normalize_points

ROOT = Path(__file__).resolve().parent


def font(size):
    path = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def draw_correspondences(pair, destination):
    """Draw only crops, points and lines; no projective image warp is performed."""
    canvas = Image.new('RGB', (2000, 1080), '#ece8e6')
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), f"A.2   {pair['source']}  ->  {pair['target']}", font=font(30), fill='#111')
    draw.text((28, 60), 'Orange: 12 fit pairs    Cyan: 4 check pairs (excluded from fitting)', font=font(23), fill='#4e4b49')
    locations = [[], []]
    for panel, key in enumerate(('source', 'target')):
        with Image.open(ROOT.parent/'A1'/'images'/pair[key]) as photo:
            points = np.array([p[key] for p in pair['points']])
            x0, y0 = np.maximum(0, np.floor(points.min(axis=0)-200)).astype(int)
            x1, y1 = np.minimum(photo.size, np.ceil(points.max(axis=0)+200)).astype(int)
            crop = photo.crop((int(x0), int(y0), int(x1), int(y1)))
            crop.thumbnail((900, 820), Image.Resampling.LANCZOS)
            scale_x, scale_y = crop.width/(x1-x0), crop.height/(y1-y0)
            ox, oy = 30+panel*1040+(900-crop.width)//2, 150+(820-crop.height)//2
            canvas.paste(crop, (int(ox), int(oy)))
            draw.text((30+panel*1040, 112), pair[key], font=font(25), fill='#111')
            draw.text((30+panel*1040, 997), f'Original pixel crop: ({x0}, {y0}) to ({x1}, {y1})', font=font(20), fill='#4e4b49')
            for point in pair['points']:
                x, y = point[key]
                locations[panel].append((float(ox+(x-x0)*scale_x), float(oy+(y-y0)*scale_y)))
    # Thin links retain individual numbered identities across the two crops.
    for index, point in enumerate(pair['points']):
        color = '#ea580c' if point['role']=='fit' else '#0891b2'
        a, b = locations[0][index], locations[1][index]
        draw.line((a, b), fill=color, width=1)
    for panel in (0, 1):
        for index, point in enumerate(pair['points']):
            x, y = locations[panel][index]
            color = '#fb923c' if point['role']=='fit' else '#22d3ee'
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline='black', width=1)
            # Alternating label offsets keep close frame-corner pairs legible.
            dx, dy = (8, -30) if index % 2 == 0 else (8, 7)
            draw.text((x+dx, y+dy), str(point['id']), font=font(22), fill=color, stroke_width=2, stroke_fill='#111')
    canvas.save(destination, quality=96, subsampling=0)


def patch_sheet(pair, destination):
    """Save close-up pairs so repeated bays/floors can be audited visually."""
    canvas = Image.new('RGB', (1600, 1080), 'white')
    draw = ImageDraw.Draw(canvas)
    for panel_name in ('source', 'target'):
        with Image.open(ROOT.parent/'A1'/'images'/pair[panel_name]) as photo:
            side = 0 if panel_name=='source' else 1
            for index, point in enumerate(pair['points']):
                ox, oy = (index%4)*400, (index//4)*270
                if side == 0:
                    draw.text((ox+4, oy+2), f"{point['id']} ({point['role']}) NCC {point['ncc']:.2f}", font=font(16), fill='black')
                x, y = point[panel_name]
                patch = photo.crop((round(x)-60, round(y)-60, round(x)+60, round(y)+60)).resize((190, 190))
                pd = ImageDraw.Draw(patch)
                pd.line((85, 95, 105, 95), fill='red', width=1)
                pd.line((95, 85, 95, 105), fill='red', width=1)
                canvas.paste(patch, (ox+side*198, oy+30))
                draw.text((ox+side*198, oy+225), f'{panel_name} {x:.0f},{y:.0f}', font=font(16), fill='black')
    canvas.save(destination, quality=95)


def main():
    data = json.loads((ROOT/'correspondences.json').read_text(encoding='utf-8'))
    (ROOT/'figures').mkdir(exist_ok=True)
    (ROOT/'equations').mkdir(exist_ok=True)
    results = {'coordinate_convention': data['coordinate_convention'],
               'matrix_convention': 'p_target ~ H_source_to_target @ [x_source, y_source, 1]; h33=1',
               'fitting': 'Normalized eight-parameter linear least squares; 12 fit points, no outlier rejection',
               'validation_note': 'Four excluded check points per pair. Annotations were visually revised during QA; this is not a blind validation set or a global image-warp test.',
               'pairs': []}
    sections, summary_rows = [], []
    for pair in data['pairs']:
        if pair.get('status') == 'insufficient_overlap':
            figure = f"figures/{pair['id']}_overlap_gap.jpg"
            canvas = Image.new('RGB', (2000, 1080), '#ece8e6')
            draw = ImageDraw.Draw(canvas)
            draw.text((28, 18), f"A.2   {pair['source']} -> {pair['target']}   /   INSUFFICIENT VERIFIED OVERLAP", font=font(25), fill='#111')
            for i, key in enumerate(('source','target')):
                with Image.open(ROOT.parent/'A1'/'images'/pair[key]) as photo:
                    photo.thumbnail((900, 900))
                    canvas.paste(photo, (int(40+i*1000+(900-photo.width)//2), 110))
                draw.text((40+i*1000, 65), pair[key], font=font(25), fill='#111')
            canvas.save(ROOT/figure,quality=95)
            record = {'id':pair['id'],'source':pair['source'],'target':pair['target'],
                      'status':'insufficient_overlap','reason':pair['reason'],
                      'H_source_to_target':None,'fit_count':0,'check_count':0,'figure':figure}
            results['pairs'].append(record)
            sections.append(f'''<section class="report-section" id="pair-{pair['id']}"><p class="eyebrow">PAIR {len(results['pairs']):02d} / UNRESOLVED</p><h2>{pair['source']} → {pair['target']}</h2><div class="warning"><strong>No verified homography.</strong> {html.escape(pair['reason'])}</div><a href="{figure}"><img src="{figure}" alt="Two views with insufficient verified overlap" loading="lazy"></a><p>The visible common region is a narrow edge strip. Many candidate patches lie on repeated horizontal frames or reflections that change with viewpoint. Trial annotations were inspected and retained in the inspection folder; they are excluded from the solved correspondence data. No matrix, equation system, or invented accuracy is reported for this pair.</p></section>''')
            summary_rows.append(f"<tr><td><a href='#pair-{pair['id']}'>Pair {len(results['pairs'])} · unresolved</a></td><td>—</td><td>—</td><td>—</td></tr>")
            print(pair['id'], 'UNRESOLVED: need intermediate overlapping views')
            continue
        source = np.array([p['source'] for p in pair['points']], dtype=float)
        target = np.array([p['target'] for p in pair['points']], dtype=float)
        fit = np.array([p['role']=='fit' for p in pair['points']])
        if fit.sum() != 12 or (~fit).sum() != 4:
            raise ValueError('Each pair must have 12 fit correspondences and 4 check correspondences.')
        if any(p['search_boundary'] for p in pair['points']):
            raise ValueError(f"Unreviewed boundary match in {pair['id']}; recheck the annotation.")
        H = computeH(source[fit], target[fit])
        errors = np.linalg.norm(project_points(source, H)-target, axis=1)
        A, b = build_system(source[fit], target[fit])
        src_n, T1 = normalize_points(source[fit])
        dst_n, T2 = normalize_points(target[fit])
        An, bn = build_system(src_n, dst_n)
        Hn = T2 @ H @ np.linalg.inv(T1)
        Hn /= Hn[2, 2]
        eq_path = f"equations/{pair['id']}.json"
        equations = {'unknown_order': ['h11','h12','h13','h21','h22','h23','h31','h32'],
                     'fit_point_ids': [p['id'] for p in pair['points'] if p['role']=='fit'],
                     'A_raw': A.tolist(), 'b_raw': b.tolist(), 'A_normalized': An.tolist(),
                     'b_normalized': bn.tolist(), 'T_source': T1.tolist(), 'T_target': T2.tolist(),
                     'H_normalized': Hn.tolist(), 'rank_normalized': int(np.linalg.matrix_rank(An)),
                     'condition_raw': float(np.linalg.cond(A)), 'condition_normalized': float(np.linalg.cond(An)),
                     'normalized_algebraic_rmse': float(np.sqrt(np.mean((An @ Hn.ravel()[:8]-bn)**2)))}
        (ROOT/eq_path).write_text(json.dumps(equations, indent=2), encoding='utf-8')
        with (ROOT/'equations'/f"{pair['id']}_raw.csv").open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(equations['unknown_order']+['b'])
            writer.writerows(np.column_stack((A, b)))
        figure = f"figures/{pair['id']}_correspondences.jpg"
        closeups = f"figures/{pair['id']}_patches.jpg"
        draw_correspondences(pair, ROOT/figure)
        patch_sheet(pair, ROOT/closeups)
        record = {'id': pair['id'], 'source': pair['source'], 'target': pair['target'], 'status':'solved',
                  'source_size':pair['source_size'],'target_size':pair['target_size'],
                  'H_source_to_target': H.tolist(), 'fit_count': int(fit.sum()), 'check_count': int((~fit).sum()),
                  'forward_errors_px': errors.tolist(), 'fit_rmse_px': float(np.sqrt(np.mean(errors[fit]**2))),
                  'check_rmse_px': float(np.sqrt(np.mean(errors[~fit]**2))),
                  'fit_max_px': float(errors[fit].max()), 'check_max_px': float(errors[~fit].max()),
                  'equations_file': eq_path, 'figure': figure, 'closeups': closeups}
        results['pairs'].append(record)
        matrix = '\n'.join('  '.join(f'{entry: .10g}' for entry in row) for row in H)
        point_rows = ''.join(f"<tr><td>{p['id']}</td><td>{p['role']}</td><td>{s[0]:.2f}, {s[1]:.2f}</td><td>{t[0]:.2f}, {t[1]:.2f}</td><td>{e:.3f}</td><td>{p['ncc']:.3f}</td></tr>" for p,s,t,e in zip(pair['points'],source,target,errors))
        example = '\n'.join('  '.join(f'{v:.5g}' for v in row) for row in A[:2])
        sections.append(f'''<section class="report-section" id="pair-{pair['id']}">
<p class="eyebrow">PAIR {len(results['pairs']):02d}</p><h2>{html.escape(pair['source'])} → {html.escape(pair['target'])}</h2>
<p>H maps original source to target coordinates. Orange: fitting points 1–12. Cyan: check points 13–16.</p>
<a href="{figure}"><img src="{figure}" alt="Numbered architectural correspondences for {pair['id']}" loading="lazy"></a>
<div class="metrics"><div><strong>{record['fit_rmse_px']:.3f} px</strong><span>Fit RMSE · 12 points</span></div><div><strong>{record['check_rmse_px']:.3f} px</strong><span>Check RMSE · 4 excluded points</span></div><div><strong>{record['check_max_px']:.3f} px</strong><span>Largest check error</span></div></div>
<h3>Recovered homography</h3><pre>{matrix}</pre>
<p>Normalized system: 24 equations, 8 unknowns, rank {equations['rank_normalized']}. Condition number changes from {equations['condition_raw']:.3g} before normalization to {equations['condition_normalized']:.3g} after normalization.</p>
<details><summary>First two rows of the raw system</summary><pre>{example}</pre><p>Right-hand side: {b[0]:.4f}, {b[1]:.4f}. The normalized system is solved; the raw system is shown for reference.</p></details>
<p><a href="{eq_path}">Full raw and normalized systems (JSON)</a> · <a href="equations/{pair['id']}_raw.csv">Raw equations (CSV)</a> · <a href="{closeups}">Audit all 16 local image patches</a></p>
<details><summary>All point coordinates and errors</summary><div class="scroll"><table><thead><tr><th>Point</th><th>Role</th><th>Source (x, y)</th><th>Target (u, v)</th><th>Error (px)</th><th>Local NCC</th></tr></thead><tbody>{point_rows}</tbody></table></div></details>
</section>''')
        summary_rows.append(f"<tr><td><a href='#pair-{pair['id']}'>Pair {len(results['pairs'])}</a></td><td>{record['fit_rmse_px']:.3f}</td><td>{record['check_rmse_px']:.3f}</td><td>{record['check_max_px']:.3f}</td></tr>")
        print(pair['id'], f"fit RMSE={record['fit_rmse_px']:.3f}px; check RMSE={record['check_rmse_px']:.3f}px; max check={record['check_max_px']:.3f}px")
    results['solved_count'] = sum(p['status']=='solved' for p in results['pairs'])
    results['unresolved_count'] = len(results['pairs'])-results['solved_count']
    (ROOT/'homographies.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    report = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A.2 · Recovering Homographies</title>
<link rel="stylesheet" href="../style.css"></head><body class="report-page"><main class="shell">
<header class="site-header"><a class="brand" href="../index.html" aria-label="Stitching Photo Mosaics home">Stitching Photo<br>Mosaics</a><nav class="header-nav" aria-label="Report navigation"><a href="../index.html">Project home <span>01</span></a><a href="#method">Method <span>02</span></a><a href="#results">Measurements <span>03</span></a></nav></header>
<section class="project-hero"><p class="eyebrow">SYDE 671 · Assignment 3 · James Xie</p><h1>A.2 / RECOVERING<br>HOMOGRAPHIES</h1><p class="project-deck">Eight adjacent image pairs, an explicit overdetermined linear system, and a homography solver implemented from scratch.</p><div class="project-hero-rule"><span>Part A · Recover homographies</span><span>9 photographs · 8 verified transformations</span></div></section>
<section class="report-section" id="method"><h2>Method and correspondence provenance</h2><p>For each of eight pairs, the assistant visually selected 16 architectural points: 12 for fitting and four excluded check points. Frames were preferred over vegetation and reflections.</p><p>Points are <strong>visually seeded and locally refined</strong> on 1008 × 1344 previews, not user mouse clicks. A scaled 31 × 31 patch is matched within ±32 preview pixels, with a subpixel peak estimate. Patch scale is visually estimated per pair. Original coordinates use (p + 0.5) × original_size / preview_size − 0.5, with each image sized 3024 × 4032 or 4284 × 5712. Previews aid annotation only. OpenCV handles image loading and local correlation; H is solved by handwritten code without automatic feature matching or RANSAC.</p><p>Seeds, refined coordinates, scores and annotation corrections are saved. All fitting points have equal weight; none is removed by its H error.</p><div class="note">The A.1 originals now contain the nine selected photographs supplied for this revision. Pixel convention: (0, 0) is the center of the top-left pixel; x increases rightward and y downward. Each displayed H maps the first image into the second image, and H[2,2] = 1.</div></section>
<section class="report-section"><h2>From point pairs to a linear system</h2><p>For a source point (x, y) and its target (u, v), the projective relation p′ ∼ Hp gives:</p><div class="formula">u = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
v = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)

[x y 1 0 0 0 -u*x -u*y] h = u
[0 0 0 x y 1 -v*x -v*y] h = v

h = [h11 h12 h13 h21 h22 h23 h31 h32]ᵀ</div><p>Twelve pairs give a 24 × 8 system. Each point set is centered and scaled to mean distance √2. NumPy least squares minimizes ||Aₙhₙ − bₙ||², then H returns to original coordinates:</p><div class="formula">pₙ = T_source p        p′ₙ = T_target p′
H = inverse(T_target) · Hₙ · T_source
H = H / H[2,2]</div><p>Normalization improves conditioning. This minimizes algebraic error, not geometric reprojection error. Invalid, nonfinite, degenerate, rank-deficient or singular cases are rejected.</p><p>Forward geometric error is measured independently in original target pixels:</p><div class="formula">eᵢ = ||project(H · [xᵢ, yᵢ, 1]ᵀ) − [uᵢ, vᵢ]ᵀ||₂
RMSE = sqrt(mean(eᵢ²))</div></section>
<section class="report-section" id="results"><h2>Measured errors</h2><p>Each row describes the corresponding adjacent pair below. Smaller errors indicate closer agreement with the annotated points.</p><table><thead><tr><th>Pair</th><th>Fit RMSE (px)</th><th>Check RMSE (px)</th><th>Max check (px)</th></tr></thead><tbody>''' + ''.join(summary_rows) + '''</tbody></table><div class="warning">The check points were excluded from matrix fitting, but annotation mistakes were visually corrected during QA. They are not a blind validation set. Four check points do not establish alignment across the entire image, and no image-warp or panorama quality is claimed by A.2.</div></section>''' + '\n'.join(sections) + '''
<section class="report-section"><h2>Limitations and next stage</h2><p>Facade points support local alignment. Other depths, reflections, lens distortion and backlighting can still disagree. A fixed camera center is unverified.</p><p>IMG_8448 connects IMG_8447 → IMG_8448 → IMG_8450, completing the nine-photo chain. Low local point errors do not guarantee alignment at every depth. A.3 and A.4 show warping and mosaicing.</p><p>A.2 recovers H. See A.3 for resampling and A.4 for blending.</p></section>
<section class="report-section"><h2>Reproduce and inspect</h2><p>Run from A2 using the existing Python, NumPy, Pillow and OpenCV environment.</p><pre>python prepare_correspondences.py
python build_report.py
python -m unittest test_homography test_refinement test_mixed_resolution test_report -v</pre><p><a href="homography.py">Handwritten homography solver</a> · <a href="manual_seeds.json">Visual seeds and annotation revisions</a> · <a href="correspondences.json">Final point data</a> · <a href="homographies.json">All matrices and error measurements</a> · <a href="../A1/selected_photos.jpg">Nine original-view source previews</a></p><p>Checks cover known H values, four-point and noisy fits, large offsets, invalid inputs and local refinement. Real-photo checks verify matrices, equations, figures and point errors.</p></section>
<footer class="site-footer"><p>Assignment source: <a href="../references/assignment-3.pdf">SYDE 671 Assignment 3, A.2</a>. All source photographs are the existing local images selected in A.1. No synthetic imagery was used.</p><span>SYDE 671 · Assignment 3 · James Xie</span></footer></main></body></html>'''
    (ROOT/'report.html').write_text(report, encoding='utf-8')
    readme = '''# A.2 - Recovering Homographies

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

'''
    for r in results['pairs']:
        if r['status']=='solved':
            readme += f"- {r['source']} -> {r['target']}: fit RMSE {r['fit_rmse_px']:.3f} px; excluded-check RMSE {r['check_rmse_px']:.3f} px; maximum check error {r['check_max_px']:.3f} px.\n"
        else:
            readme += f"- {r['source']} -> {r['target']}: UNRESOLVED. {r['reason']}\n"
    (ROOT/'README.md').write_text(readme, encoding='utf-8')


if __name__ == '__main__':
    main()
