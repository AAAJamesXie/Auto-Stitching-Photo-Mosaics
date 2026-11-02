"""Generate B2 data and spatially chosen examples from unchanged B1 assets."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from descriptors import extract_descriptors
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
NAMES=['8443','8444','8445','8446','8447','8448','8450']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def choose_examples(xy,valid):
    chosen=[];pool=np.flatnonzero(valid)
    for y in [1600*.28,1600*.72]:
        for x in [1200*.2,1200*.5,1200*.8]:
            available=pool[~np.isin(pool,chosen)]
            if not len(available):return np.array(chosen,int)
            distances=np.sum((xy[available]-[x,y])**2,axis=1)
            chosen.append(int(available[np.argmin(distances)]))
    return np.array(chosen,int)

def tile(array,normalized=False):
    # Fixed display scales: no independent per-patch contrast stretch.
    data=np.clip((array+3)/6,0,1) if normalized else np.clip(array,0,1)
    return Image.fromarray(np.rint(data*255).astype(np.uint8)).convert('RGB').resize((120,120),Image.Resampling.NEAREST)

def main():
    for name in ['data','examples','figures','inspection']:(ROOT/name).mkdir(exist_ok=True)
    tracked=[PROJECT/'B1/measurements.json',PROJECT/'B1/corners.py',PROJECT/'A2/homographies.json',PROJECT/'A3/warp.py',PROJECT/'A4/results/panorama.png']
    for n in NAMES:tracked.extend([PROJECT/f'B1/working/IMG_{n}.png',PROJECT/f'B1/points/IMG_{n}.npz',PROJECT/f'A1/images/IMG_{n}.jpg'])
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in tracked};records=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for n in NAMES:
        start=time.perf_counter();path=PROJECT/f'B1/working/IMG_{n}.png'
        with Image.open(path) as im:working=im.convert('RGB')
        gray=(np.array(working,dtype=float)/255)@np.array([.299,.587,.114])
        with np.load(PROJECT/f'B1/points/IMG_{n}.npz',allow_pickle=False) as points:
            selected=points['anms_indices'];xy=points['xy_working'][selected];original=points['xy_original'][selected]
            H=points['H_original_to_working'];candidate_ids=selected.copy()
        result=extract_descriptors(gray,xy);valid=result['valid'];ids=np.array([f'IMG_{n}:{i:03d}' for i in range(len(xy))])
        data={k:result[k] for k in ['sampled8','mean','std','in_bounds','valid','descriptors']}
        data.update(feature_ids=ids,b1_rows=np.arange(len(xy)),candidate_indices=candidate_ids,xy_working=xy,xy_original=original,H_original_to_working=H)
        np.savez_compressed(ROOT/f'data/IMG_{n}.npz',**data)
        metadata=[]
        for i in range(len(xy)):
            reason='valid' if valid[i] else ('outside_support' if not result['in_bounds'][i] else 'low_variance')
            metadata.append({'feature_id':str(ids[i]),'b1_row':i,'candidate_index':int(candidate_ids[i]),'working_xy':xy[i].tolist(),'original_xy':original[i].tolist(),'valid':bool(valid[i]),'status':reason,'sample_mean':float(result['mean'][i]),'sample_std':float(result['std'][i])})
        (ROOT/f'data/IMG_{n}.json').write_text(json.dumps({'source':f'IMG_{n}.jpg','working_image':f'../../B1/working/IMG_{n}.png','descriptor_shape':[len(xy),64],'rows':metadata},indent=2,allow_nan=False),encoding='utf-8')
        examples=choose_examples(xy,valid)
        np.savez_compressed(ROOT/f'examples/IMG_{n}.npz',b1_rows=examples,raw40=result['raw40'][examples],blurred40=result['blurred40'][examples],sampled8=result['sampled8'][examples],normalized8=result['descriptors'][examples].reshape(-1,8,8))
        location=working.copy();draw=ImageDraw.Draw(location)
        for label,i in enumerate(examples):
            x,y=xy[i];draw.rectangle((x-20,y-20,x+20,y+20),outline='#ff4b00',width=3)
            draw.ellipse((x-3,y-3,x+3,y+3),fill='#00e5ff');draw.text((x+24,y-25),f'{label+1} / {i}',font=font,fill='#ff4b00',stroke_width=2,stroke_fill='black')
        location.save(ROOT/f'figures/IMG_{n}_locations.jpg',quality=96)
        sheet=Image.new('RGB',(1800,640),'#ece8e6');draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(examples):
            x=(j%3)*600+20;y=(j//3)*320+10
            draw.text((x,y),f'{j+1} / B1 row {i} · candidate {candidate_ids[i]}',font=font,fill='black')
            variants=[result['raw40'][i],result['blurred40'][i],result['sampled8'][i],result['descriptors'][i].reshape(8,8)]
            for k,(a,label) in enumerate(zip(variants,['Raw 40×40','Blurred 40×40','Sampled 8×8','Normalized'])):
                sheet.paste(tile(a,k==3),(x+k*140,y+65));draw.text((x+k*140,y+40),label,font=small,fill='black')
            draw.text((x,y+200),f'mean {result["mean"][i]:.4f} · std {result["std"][i]:.4f}',font=small,fill='black')
            draw.text((x,y+226),'Normalized display: −3 to +3, zero = middle gray',font=small,fill='black')
        sheet.save(ROOT/f'figures/IMG_{n}_examples.jpg',quality=97)
        # Data-level contract checks on the actual saved order.
        assert len(xy)==500 and np.isfinite(result['descriptors']).all()
        np.testing.assert_allclose(result['descriptors'][valid].mean(axis=1),0,atol=1e-12)
        np.testing.assert_allclose(result['descriptors'][valid].std(axis=1),1,atol=1e-12)
        assert np.all(result['descriptors'][~valid]==0)
        record={'image':f'IMG_{n}.jpg','input_points':len(xy),'valid_count':int(valid.sum()),'invalid_support':int((~result['in_bounds']).sum()),'invalid_low_variance':int((result['in_bounds']&~valid).sum()),'descriptor_shape':list(result['descriptors'].shape),'std_min_in_bounds':float(result['std'][result['in_bounds']].min()),'max_abs_valid_descriptor_mean':float(np.abs(result['descriptors'][valid].mean(axis=1)).max()) if valid.any() else None,'max_abs_valid_descriptor_std_error':float(np.abs(result['descriptors'][valid].std(axis=1)-1).max()) if valid.any() else None,'example_b1_rows':examples.tolist(),'seconds_including_io':time.perf_counter()-start}
        records.append(record);print(json.dumps(record),flush=True)
    assert all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    params={'gray':'RGB/255 dot (.299,.587,.114)','patch_size':40,'patch_center_offsets':[-19.5,19.5],'blur_sigma':2.,'blur_kernel':[13,13],'blur_boundary':'OpenCV REFLECT_101 at working-image boundary; smooth whole image','sample_region_indices':[2,7,12,17,22,27,32,37],'sample_world_offsets':[-17.5,-12.5,-7.5,-2.5,2.5,7.5,12.5,17.5],'normalization':'population std ddof0; valid iff support inside and std>1e-6; no clipping','flatten_order':'C row-major','std_threshold':1e-6,'rotation':'axis aligned, no rotation','display_normalized_range':[-3,3]}
    (ROOT/'measurements.json').write_text(json.dumps({'parameters':params,'records':records,'input_sha256':hashes},indent=2,allow_nan=False),encoding='utf-8')
    print('Seven descriptor datasets saved; upstream inputs unchanged.',flush=True)

if __name__=='__main__':main()
