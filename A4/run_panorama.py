"""Seven original photos, one 60%-scale destination; handwritten sampling/blend."""
import json,sys,time,gc
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'A3'))
from warp import output_canvas,warpImageBilinear
from blend import feather_weights,blend_layers
from panorama_geometry import compose_transforms
from run_mosaic import sha,rgb8,save_rgba,plate

NAMES=['8443','8444','8445','8446','8447','8448','8450']
COLORS=['#d94735','#dd8c18','#b1a516','#369264','#268fa4','#5165c4','#b75ca2']

def presentation(rgb,alpha):
    im=Image.fromarray(rgb8(rgb));im.paste('#ece8e6',(0,0),Image.fromarray((~alpha).astype(np.uint8)*255));return im

def main():
    start=time.perf_counter()
    for name in ['results','figures','inspection']:(ROOT/name).mkdir(exist_ok=True)
    inputs=[PROJECT/'A1/images'/f'IMG_{n}.jpg' for n in NAMES]
    tracked=inputs+[PROJECT/'A2/homographies.json',PROJECT/'A2/homography.py',PROJECT/'A3/warp.py',ROOT/'blend.py']
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in tracked}
    pairs=json.loads((PROJECT/'A2/homographies.json').read_text(encoding='utf-8'))['pairs']
    lookup={p['id']:np.array(p['H_source_to_target']) for p in pairs}
    transforms=compose_transforms(NAMES,3,lookup,.6)
    sizes=[Image.open(p).size for p in inputs]
    boxes=[output_canvas((h,w),H) for (w,h),H in zip(sizes,transforms)]
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes)
    x1=max(b[0]+b[2]-1 for b in boxes);y1=max(b[1]+b[3]-1 for b in boxes)
    canvas=(x0,y0,x1-x0+1,y1-y0+1);_,_,w,h=canvas
    coverage=np.zeros((h,w),np.uint8);denominator=np.zeros((h,w),np.float32)
    baseline=np.zeros((h,w,3),np.uint8)
    # Fixed distributed probes, plus verification of every sole-support pixel later.
    probes=[(int(x),int(y)) for y in np.linspace(h*.2,h*.8,3) for x in np.linspace(w*.06,w*.94,15)]
    samples=[[] for _ in probes];records=[];T=np.array([[1.,0,-x0],[0,1,-y0],[0,0,1]])
    def layers():
        for i,(path,H,size,box) in enumerate(zip(inputs,transforms,sizes,boxes)):
            tick=time.perf_counter();source=np.array(Image.open(path).convert('RGB'))
            warped=warpImageBilinear(source,H,canvas=canvas)
            wr=warpImageBilinear(feather_weights(source.shape),H,canvas=canvas)
            weight=wr.image;weight[~warped.alpha]=0;del wr,source
            valid=warped.alpha;coverage[:]+=valid;denominator[:]+=weight
            baseline[valid]=rgb8(warped.image[valid])
            Image.fromarray(valid.astype(np.uint8)*255).save(ROOT/f'results/panorama_{NAMES[i]}_mask.png')
            Image.fromarray(np.rint(weight*65535).astype(np.uint16)).save(ROOT/f'results/panorama_{NAMES[i]}_weight.png')
            for j,(x,y) in enumerate(probes):samples[j].append({'image':path.name,'rgb':warped.image[y,x].tolist(),'weight':float(weight[y,x])})
            records.append({'image':path.name,'original_size':list(size),'H_original_to_scaled_reference':H.tolist(),'H_original_to_canvas':(T@H).tolist(),'bounds_scaled_reference':list(box),'valid_pixels':int(valid.sum()),'warp_seconds_including_mask_io':time.perf_counter()-tick})
            print(f'{i+1}/7 {path.name}: {valid.sum():,} valid pixels',flush=True)
            yield warped.image,weight
            del warped,weight,valid;gc.collect()
    blended,alpha=blend_layers(layers());assert np.array_equal(alpha,coverage>0)
    assert np.array_equal(rgb8(blended)[coverage==1],baseline[coverage==1])
    save_rgba(blended,alpha,ROOT/'results/panorama.png');save_rgba(baseline,alpha,ROOT/'results/panorama_overwrite.png')
    Image.fromarray(alpha.astype(np.uint8)*255).save(ROOT/'results/panorama_union_mask.png')
    Image.fromarray(coverage).save(ROOT/'results/panorama_coverage_count.png')
    # Contribution display uses 16-bit stored source weights; numerical RGB uses float32.
    diagram=Image.new('RGB',(1800,4*460+30),'#ece8e6');draw=ImageDraw.Draw(diagram);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    sum_quantized=np.zeros((h,w),np.float32)
    for n in NAMES:sum_quantized+=np.array(Image.open(ROOT/f'results/panorama_{n}_weight.png'),dtype=np.float32)
    for i,n in enumerate(NAMES):
        raw=np.array(Image.open(ROOT/f'results/panorama_{n}_weight.png'),dtype=np.float32)
        norm=np.divide(raw,sum_quantized,out=np.zeros_like(raw),where=sum_quantized>0)
        # Dark contribution on the shared gray canvas; all seven rows use same scale.
        display=np.rint(236-210*norm).astype(np.uint8)
        im=Image.fromarray(display).convert('RGB');im.thumbnail((840,410),Image.Resampling.LANCZOS)
        cx=(i%2)*900;cy=(i//2)*460
        diagram.paste(im,(cx+30,cy+55));draw.text((cx+30,cy+15),f'{i+1:02d} / IMG_{n} contribution',font=font,fill='black')
    diagram.save(ROOT/'figures/panorama_contributions.jpg',quality=95)
    overview=presentation(blended,alpha);overview.thumbnail((2200,1400));overview.save(ROOT/'figures/panorama.jpg',quality=96)
    count_palette=np.array([[236,232,230],[212,70,45],[229,164,56],[115,165,105],[50,147,172],[80,100,190],[150,90,180],[60,50,70]],np.uint8)
    im=Image.fromarray(count_palette[coverage]);im.thumbnail((2000,1300));im.save(ROOT/'figures/panorama_coverage.jpg',quality=96)
    # Presentation pair is stacked to keep the seven-view horizontal panorama readable.
    comparison=Image.new('RGB',(2200,2*overview.height+110),'#ece8e6');d=ImageDraw.Draw(comparison)
    for i,(a,label) in enumerate([(baseline,'Direct overwrite / later photos on top'),(blended,'Normalized feather blending / same seven views')]):
        im=presentation(a,alpha);im.thumbnail((2200,1400));yy=i*(overview.height+55);d.text((18,yy+12),label,font=font,fill='black');comparison.paste(im,(0,yy+50))
    comparison.save(ROOT/'figures/panorama_comparison.jpg',quality=95)
    detail_boxes=[(2750,1100,2990,1340),(4600,1550,4840,1790)]
    for i,box in enumerate(detail_boxes):
        ims=[Image.fromarray(rgb8(a)).crop(box).resize((720,720),Image.Resampling.NEAREST) for a in [baseline,blended]]
        plate(ims,['Overwrite / same output crop','Feather / pixels replicated 3x'],ROOT/f'figures/panorama_detail_{i+1}.jpg',height=740)
    probe_records=[]
    for (x,y),terms in zip(probes,samples):
        denom=sum(t['weight'] for t in terms)
        expected=sum((np.array(t['rgb'])*t['weight'] for t in terms),start=np.zeros(3))/denom if denom else np.zeros(3)
        np.testing.assert_allclose(blended[y,x],expected,atol=6e-5,rtol=1e-6)
        probe_records.append({'canvas_xy':[x,y],'coverage':int(coverage[y,x]),'expected_float64_rgb':expected.tolist(),'actual_float32_rgb':blended[y,x].tolist(),'terms':terms})
    assert all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    data={'images':[p.name for p in inputs],'reference':'IMG_8446.jpg','destination_scale':.6,'scale_rule':'original source centers sampled directly into .6*reference coordinates +(.6-1)/2; no pre-resized sources','canvas':list(canvas),'layers':records,'valid_pixels':int(alpha.sum()),'overlap_pixels':int((coverage>=2).sum()),'sole_support_pixels':int((coverage==1).sum()),'coverage_histogram':{str(i):int((coverage==i).sum()) for i in range(8)},'detail_canvas_boxes':detail_boxes,'blend_order':NAMES,'weight_rule':'positive normalized source rectangular edge distance, bilinear warp, zero outside valid domain; online normalized weighted mean','weight_png_note':'16-bit quantization for visualization only; full numerical blending uses float32 weights','input_sha256':hashes,'scalar_checks':probe_records,'elapsed_seconds_including_io':time.perf_counter()-start}
    (ROOT/'panorama_measurements.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:data[k] for k in ['canvas','valid_pixels','overlap_pixels','elapsed_seconds_including_io']}),flush=True)

if __name__=='__main__':main()
