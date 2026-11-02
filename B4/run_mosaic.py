"""One automatic mosaic plus same-input manual baseline, on identical canvas."""
import gc,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'A3'));sys.path.insert(0,str(PROJECT/'A4'))
from warp import output_canvas,warpImageBilinear
from blend import feather_weights,blend_layers
from panorama_geometry import compose_transforms
from run_panorama import presentation
NAMES=['8443','8444','8445','8446','8447','8448','8450']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rgb8(a):return np.clip(np.rint(a),0,255).astype(np.uint8)
def save_rgba(a,mask,path):
    out=np.empty((*mask.shape,4),np.uint8);out[:,:,:3]=rgb8(a);out[:,:,3]=mask.astype(np.uint8)*255;Image.fromarray(out).save(path)

def render(label,transforms,canvas):
    start=time.perf_counter();x0,y0,w,h=canvas
    count=np.zeros((h,w),np.uint8);overwrite=np.zeros((h,w,3),np.uint8)
    probes=[(int(x),int(y)) for y in np.linspace(.2*h,.8*h,3) for x in np.linspace(.06*w,.94*w,15)]
    terms=[[] for _ in probes];layers_data=[];T=np.array([[1.,0,-x0],[0,1,-y0],[0,0,1]])
    def layers():
        for n,H in zip(NAMES,transforms):
            with Image.open(PROJECT/f'A1/images/IMG_{n}.jpg') as im:source=np.array(im.convert('RGB'))
            warped=warpImageBilinear(source,H,canvas=canvas)
            weight=warpImageBilinear(feather_weights(source.shape),H,canvas=canvas).image;weight[~warped.alpha]=0
            del source
            count[:]+=warped.alpha;overwrite[warped.alpha]=rgb8(warped.image[warped.alpha])
            Image.fromarray(warped.alpha.astype(np.uint8)*255).save(ROOT/f'results/{label}_{n}_mask.png')
            for i,(x,y) in enumerate(probes):terms[i].append({'image':n,'rgb':warped.image[y,x].tolist(),'weight':float(weight[y,x])})
            layers_data.append({'image':f'IMG_{n}.jpg','H_original_to_scaled_reference':H.tolist(),'H_original_to_canvas':(T@H).tolist(),'valid_pixels':int(warped.alpha.sum())})
            print(label,n,'warped',flush=True)
            yield warped.image,weight
            del warped,weight;gc.collect()
    rgb,alpha=blend_layers(layers());assert np.array_equal(alpha,count>0)
    assert np.array_equal(rgb8(rgb)[count==1],overwrite[count==1])
    checks=[]
    for (x,y),values in zip(probes,terms):
        denominator=sum(v['weight'] for v in values)
        expected=sum((np.array(v['rgb'])*v['weight'] for v in values),start=np.zeros(3))/denominator if denominator else np.zeros(3)
        np.testing.assert_allclose(rgb[y,x],expected,atol=6e-5,rtol=1e-6)
        checks.append({'canvas_xy':[x,y],'expected_float64_rgb':expected.tolist(),'actual_float32_rgb':rgb[y,x].tolist(),'terms':values})
    save_rgba(rgb,alpha,ROOT/f'results/{label}_mosaic.png')
    Image.fromarray(alpha.astype(np.uint8)*255).save(ROOT/f'results/{label}_union_mask.png')
    Image.fromarray(count).save(ROOT/f'results/{label}_coverage_count.png')
    preview=presentation(rgb,alpha);preview.thumbnail((2200,1400));preview.save(ROOT/f'figures/{label}_mosaic.jpg',quality=96)
    metadata={'label':label,'valid_pixels':int(alpha.sum()),'overlap_pixels':int((count>=2).sum()),'sole_support_pixels':int((count==1).sum()),'coverage_histogram':{str(i):int((count==i).sum()) for i in range(8)},'layers':layers_data,'scalar_checks':checks,'seconds_including_io':time.perf_counter()-start}
    return metadata

def main():
    for name in ['results','figures','inspection']:(ROOT/name).mkdir(exist_ok=True)
    # Automatic H was already frozen by run_ransac.py; manual H is first read here.
    automatic_path=ROOT/'automatic_homographies.json';automatic_hash=sha(automatic_path)
    automatic=json.loads(automatic_path.read_text(encoding='utf-8'))
    manual=json.loads((PROJECT/'A2/homographies.json').read_text(encoding='utf-8'))
    transforms={'automatic':compose_transforms(NAMES,3,{p['pair']:np.array(p['H_original']) for p in automatic['pairs']},.6),
                'manual':compose_transforms(NAMES,3,{p['id']:np.array(p['H_source_to_target']) for p in manual['pairs']},.6)}
    sizes=[]
    for n in NAMES:
        with Image.open(PROJECT/f'A1/images/IMG_{n}.jpg') as im:sizes.append(im.size)
    boxes=[output_canvas((h,w),H) for group in transforms.values() for (w,h),H in zip(sizes,group)]
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes);x1=max(b[0]+b[2]-1 for b in boxes);y1=max(b[1]+b[3]-1 for b in boxes)
    canvas=(x0,y0,x1-x0+1,y1-y0+1)
    if canvas[2]*canvas[3]>40_000_000:raise ValueError('Common comparison canvas exceeds approved sampler allocation limit.')
    tracked=[PROJECT/f'A1/images/IMG_{n}.jpg' for n in NAMES]+[PROJECT/'A2/homographies.json',PROJECT/'A2/homography.py',PROJECT/'A3/warp.py',PROJECT/'A4/blend.py',PROJECT/'A4/results/panorama.png',PROJECT/'A4/panorama_measurements.json',automatic_path]
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in tracked};results=[]
    print('common canvas',canvas,flush=True)
    for label,group in transforms.items():results.append(render(label,group,canvas));gc.collect()
    assert sha(automatic_path)==automatic_hash and all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    # Both previews share one numerical canvas and a single display scale.
    with Image.open(ROOT/'figures/automatic_mosaic.jpg') as im:size=im.size
    sheet=Image.new('RGB',(size[0],2*size[1]+110),'#ece8e6');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',26)
    for i,(label,title) in enumerate([('manual','Manual correspondences / A.2 transforms'),('automatic','Automatic B.1–B.4 / RANSAC transforms')]):
        y=i*(size[1]+55);draw.text((18,y+12),title,font=font,fill='black')
        with Image.open(ROOT/f'figures/{label}_mosaic.jpg') as im:sheet.paste(im,(0,y+50))
    sheet.save(ROOT/'figures/manual_vs_automatic.jpg',quality=96)
    detail_boxes=[(2750,1100,2990,1340),(4600,1550,4840,1790)]
    for i,box in enumerate(detail_boxes):
        detail=Image.new('RGB',(1800,820),'#ece8e6');draw=ImageDraw.Draw(detail)
        for j,(label,title) in enumerate([('manual','Manual / same canvas crop'),('automatic','Automatic / pixels replicated 3x')]):
            with Image.open(ROOT/f'results/{label}_mosaic.png') as im:crop=im.crop(box).convert('RGB').resize((720,720),Image.Resampling.NEAREST)
            draw.text((j*900+24,18),title,font=font,fill='black');detail.paste(crop,(j*900+90,80))
        detail.save(ROOT/f'figures/detail_{i+1}.jpg',quality=97)
    # Re-rendered manual baseline should preserve the prior A4 saved pixels at its world origin.
    old=json.loads((PROJECT/'A4/panorama_measurements.json').read_text(encoding='utf-8'));ox,oy,ow,oh=old['canvas'];dx,dy=ox-x0,oy-y0
    with Image.open(PROJECT/'A4/results/panorama.png') as im:previous=np.array(im)
    with Image.open(ROOT/'results/manual_mosaic.png') as im:current=np.array(im.crop((dx,dy,dx+ow,dy+oh)))
    assert np.array_equal(previous,current)
    data={'images':[f'IMG_{n}.jpg' for n in NAMES],'reference':'IMG_8446.jpg','scale':.6,'canvas':list(canvas),'automatic_models_sha256_before_manual_read':automatic_hash,'automatic_models_sha256_after_render':sha(automatic_path),'methods':results,'detail_canvas_boxes':detail_boxes,'manual_A4_pixel_equality':True,'input_sha256':hashes}
    (ROOT/'mosaic_measurements.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    print('One automatic mosaic and common-canvas manual comparison saved. Prior A4 pixels match exactly.',flush=True)

if __name__=='__main__':main()
