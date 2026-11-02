"""Generate real-photo A.3 experiments using only the handwritten warp samplers.

Pillow resizing below is confined to presentation thumbnails. It is never used
to calculate a projective warp, a rectification, or a benchmark result.
"""
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from warp import warpImageNearestNeighbor, warpImageBilinear, output_canvas

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'A2'))
from homography import computeH, project_points

METHODS = {'nearest':warpImageNearestNeighbor, 'bilinear':warpImageBilinear}


def rgba(result):
    array = np.empty(result.alpha.shape+(4,), dtype=np.uint8)
    array[..., :3] = np.clip(np.rint(result.image), 0, 255).astype(np.uint8)
    array[..., 3] = result.alpha.astype(np.uint8)*255
    return Image.fromarray(array)


def source_annotation(source, quad, path):
    photo = Image.fromarray(source)
    thumb = photo.copy()
    thumb.thumbnail((900, 1100))
    d = ImageDraw.Draw(thumb)
    scale = thumb.width/photo.width
    p = np.asarray(quad)*scale
    d.line([tuple(q) for q in p]+[tuple(p[0])], fill='#ff5a00', width=3)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    for i,(x,y) in enumerate(p):
        d.ellipse((x-5,y-5,x+5,y+5),fill='#ff5a00')
        d.text((x+7,y-25),str(i+1),fill='#ff5a00',font=font,stroke_width=2,stroke_fill='black')
    thumb.save(path,quality=95)


def compare_figure(paths, destination, title, magnification=1):
    canvas = Image.new('RGB',(1800,1000),'#ece8e6')
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
    d.text((25,18),title,font=font,fill='#111')
    for i,(method,path) in enumerate(paths.items()):
        with Image.open(path) as raw:
            image = raw.convert('RGBA')
            if magnification != 1:
                # Integer pixel replication reveals the saved sample grid equally.
                a=np.asarray(image)
                image=Image.fromarray(np.repeat(np.repeat(a,magnification,0),magnification,1))
            image.thumbnail((850,850),Image.Resampling.NEAREST)
            background=Image.new('RGB',image.size,'#dfdad7')
            background.paste(image,mask=image.getchannel('A'))
            canvas.paste(background,(25+i*900+(850-image.width)//2,100+(850-image.height)//2))
        d.text((25+i*900,65),method.upper(),font=font,fill='#111')
    canvas.save(destination,quality=96,subsampling=0)


def benchmark(source,H,canvas):
    x,y,w,h=canvas
    # A common centered destination ROI bounds the repeat cost on the large photos.
    bw,bh=min(512,w),min(512,h)
    roi=(x+(w-bw)//2,y+(h-bh)//2,bw,bh)
    medians, times, sampled = {}, {}, {}
    for name,method in METHODS.items():
        method(source,H,canvas=(roi[0],roi[1],min(32,bw),min(32,bh)))
        durations=[]
        for _ in range(3):
            gc.collect()
            start=time.perf_counter()
            result=method(source,H,canvas=roi)
            durations.append(time.perf_counter()-start)
            del result
        times[name]=durations
        medians[name]=float(np.median(durations))
        sampled[name]=method(source,H,canvas=roi)
    a,b=sampled['nearest'],sampled['bilinear']
    if not np.array_equal(a.alpha,b.alpha):
        raise AssertionError('Sampling methods must use the same continuous validity domain.')
    common=a.alpha
    difference=np.abs(a.image[common].astype(float)-b.image[common].astype(float))
    return {'canvas':list(roi),'destination_pixels':bw*bh,'valid_pixels':int(common.sum()),
            'runs_seconds':times,'median_seconds':medians,
            'bilinear_to_nearest_time_ratio':medians['bilinear']/medians['nearest'],
            'mean_absolute_sample_difference':float(difference.mean()) if difference.size else None,
            'max_absolute_sample_difference':float(difference.max()) if difference.size else None,
            'difference_note':'Difference between methods on the same valid ROI, before PNG quantization; not an accuracy or ground-truth error score.'}


def main():
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'figures').mkdir(exist_ok=True)
    hashes={}
    for p in [*sorted((PROJECT/'A1'/'images').glob('*.jpg')), PROJECT/'A2'/'homographies.json',PROJECT/'A2'/'homography.py']:
        hashes[str(p.relative_to(PROJECT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    annotations=json.loads((ROOT/'rectification_points.json').read_text(encoding='utf-8'))
    cases=[]
    for c in annotations['cases']:
        c=dict(c)
        c.update(kind='rectification',H=computeH(c['source_quad'],c['target_quad']).tolist())
        cases.append(c)
    previous=json.loads((PROJECT/'A2'/'homographies.json').read_text(encoding='utf-8'))
    correspondence=json.loads((PROJECT/'A2'/'correspondences.json').read_text(encoding='utf-8'))
    point_pairs={p['id']:p for p in correspondence['pairs']}
    for p in previous['pairs']:
        if p['id'] in ['8447_8448','8451_8452']:
            w,h=p['source_size']
            cases.append({'id':p['id']+'_projection','kind':'projection','image':p['source'],
                          'reference':p['target'],'label':p['source']+' into '+p['target']+' coordinates',
                          'H':p['H_source_to_target'],'canvas':list(output_canvas((h,w),p['H_source_to_target'])),
                          'detail_world_center':point_pairs[p['id']]['points'][12]['target'],
                          'A2_check_rmse_px':p['check_rmse_px']})
    records={'pixel_convention':'Integer pixel centers. H maps source to target/world; origin records the top-left world center of the returned canvas.',
             'timing_protocol':'One small ROI warm-up, three identical-ROI runs per method, median wall time. Includes input validation, allocation, inverse coordinates and sampling. Excludes image I/O, PNG encoding and presentation resizing.',
             'software':{'python':platform.python_version(),'numpy':np.__version__,'pillow':PIL.__version__},'cases':[]}
    for case in cases:
        print('START',case['id'], 'canvas',case['canvas'],flush=True)
        with Image.open(PROJECT/'A1'/'images'/case['image']) as photo:
            source=np.asarray(photo.convert('RGB'))
        H=np.array(case['H'],dtype=float)
        record=dict(case)
        record['source_size']=[int(source.shape[1]),int(source.shape[0])]
        record['benchmark']=benchmark(source,H,case['canvas'])
        if case['kind']=='rectification':
            record['corner_fit_errors_px']=np.linalg.norm(project_points(case['source_quad'],H)-case['target_quad'],axis=1).tolist()
            record['annotated_source']='figures/'+case['id']+'_source.jpg'
            source_annotation(source,case['source_quad'],ROOT/record['annotated_source'])
        outputs, details, masks = {}, {}, {}
        expected_alpha_hash=None
        for name,method in METHODS.items():
            start=time.perf_counter()
            result=method(source,H,canvas=case['canvas'])
            elapsed=time.perf_counter()-start
            alpha_hash=hashlib.sha256(result.alpha.tobytes()).hexdigest()
            if expected_alpha_hash is not None and alpha_hash!=expected_alpha_hash:
                raise AssertionError('Full-resolution validity masks disagree.')
            expected_alpha_hash=alpha_hash
            prefix='results/'+case['id']+'_'+name
            outputs[name]=prefix+'.png'
            picture=rgba(result)
            picture.save(ROOT/outputs[name],compress_level=3)
            masks[name]=prefix+'_mask.png'
            Image.fromarray(result.alpha.astype(np.uint8)*255).save(ROOT/masks[name])
            record.setdefault('full_resolution',{})[name]={'seconds_excluding_io':elapsed,
                'width':picture.width,'height':picture.height,'origin':list(result.origin),
                'valid_pixels':int(result.alpha.sum()),'alpha_sha256':alpha_hash,
                'H_source_to_canvas':result.H_source_to_canvas.tolist()}
            if case['kind']=='rectification':
                cx,cy=32,32
            else:
                cx=round(case['detail_world_center'][0]-result.origin[0])
                cy=round(case['detail_world_center'][1]-result.origin[1])
            box=(max(0,cx-20),max(0,cy-20),min(picture.width,cx+100),min(picture.height,cy+100))
            details[name]='figures/'+case['id']+'_'+name+'_detail.png'
            picture.crop(box).save(ROOT/details[name])
            record['detail_canvas_box']=list(box)
            del result,picture
            gc.collect()
        record.update(outputs=outputs,masks=masks)
        record['overview']='figures/'+case['id']+'_comparison.jpg'
        compare_figure({k:ROOT/v for k,v in outputs.items()},ROOT/record['overview'],case['label'])
        record['detail']='figures/'+case['id']+'_details.jpg'
        compare_figure({k:ROOT/v for k,v in details.items()},ROOT/record['detail'],'Same output region / pixels replicated 4x',magnification=4)
        records['cases'].append(record)
        (ROOT/'measurements.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('DONE',case['id'],record['benchmark']['median_seconds'],flush=True)
        del source
    for file,digest in hashes.items():
        if hashlib.sha256((PROJECT/file).read_bytes()).hexdigest()!=digest:
            raise AssertionError('An A1/A2 input was changed: '+file)
    print('All four experiments saved; A1/A2 input hashes unchanged.',flush=True)


if __name__=='__main__':
    main()
