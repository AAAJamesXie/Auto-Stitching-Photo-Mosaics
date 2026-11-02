"""Estimate and freeze automatic H values before any manual-H comparison."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ransac import ransac_homography,working_h_to_original,symmetric_errors
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
NAMES=['8443','8444','8445','8446','8447','8448','8450']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for name in ['data','figures','results','inspection']:(ROOT/name).mkdir(exist_ok=True)
    tracked=[PROJECT/'A2/homography.py',PROJECT/'B3/measurements.json']
    for n in NAMES:tracked.extend([PROJECT/f'B2/data/IMG_{n}.npz',PROJECT/f'B1/working/IMG_{n}.png'])
    for n,m in zip(NAMES[:-1],NAMES[1:]):tracked.append(PROJECT/f'B3/data/{n}_{m}.npz')
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in tracked};records=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    for index,(n,m) in enumerate(zip(NAMES[:-1],NAMES[1:])):
        pair=f'{n}_{m}'
        with np.load(PROJECT/f'B3/data/{pair}.npz',allow_pickle=False) as z:b3={k:z[k] for k in z.files}
        with np.load(PROJECT/f'B2/data/IMG_{n}.npz',allow_pickle=False) as z:C_source=z['H_original_to_working']
        with np.load(PROJECT/f'B2/data/IMG_{m}.npz',allow_pickle=False) as z:C_target=z['H_original_to_working']
        mutual=b3['mutual_nearest'][b3['match_source_rows']]
        src=b3['match_source_xy_working'][mutual];dst=b3['match_target_xy_working'][mutual]
        result=ransac_homography(src,dst,seed=1337+index)
        original=working_h_to_original(result['H'],C_source,C_target)
        all_errors=symmetric_errors(b3['match_source_xy_working'],b3['match_target_xy_working'],result['H'])
        input_rows=np.flatnonzero(mutual)
        np.savez_compressed(ROOT/f'data/{pair}.npz',source_working=src,target_working=dst,
                            source_b2_rows=b3['match_source_rows'][mutual],target_b2_rows=b3['match_target_rows'][mutual],
                            source_feature_ids=b3['match_source_feature_ids'][mutual],target_feature_ids=b3['match_target_feature_ids'][mutual],
                            source_original=b3['match_source_xy_original'][mutual],target_original=b3['match_target_xy_original'][mutual],
                            b3_accepted_indices=input_rows,H_working=result['H'],H_original=original,
                            symmetric_errors=result['errors'],inliers=result['inliers'],fitted_mask=result['fitted_mask'],
                            initial_consensus=result['initial_consensus'],best_sample=result['best_sample'],
                            all_b3_symmetric_errors=all_errors,all_b3_under_threshold=all_errors<=3)
        errors=result['errors'][result['inliers']]
        numeric={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        rec={'pair':pair,'input_count':len(src),'inlier_count':int(result['inliers'].sum()),'outlier_count':int((~result['inliers']).sum()),
             'median_inlier_error_working_px':float(np.median(errors)),'rmse_inlier_error_working_px':float(np.sqrt(np.mean(errors**2))),
             'max_inlier_error_working_px':float(errors.max()),'H_working':result['H'].tolist(),'H_original':original.tolist(),
             'all_b3_ratio_count':int(len(mutual)),'all_b3_under_threshold':int((all_errors<=3).sum()),'ransac':numeric}
        (ROOT/f'data/{pair}.json').write_text(json.dumps(rec,indent=2,allow_nan=False),encoding='utf-8')
        with Image.open(PROJECT/f'B1/working/IMG_{n}.png') as im:a=im.convert('RGB')
        with Image.open(PROJECT/f'B1/working/IMG_{m}.png') as im:b=im.convert('RGB')
        image=Image.new('RGB',(2520,1690),'#ece8e6');image.paste(a,(0,90));image.paste(b,(1320,90));draw=ImageDraw.Draw(image)
        draw.text((20,20),f'{pair} / {rec["inlier_count"]} inliers, {rec["outlier_count"]} outliers / symmetric RMS <= 3 px',font=font,fill='black')
        # Draw rejected model inputs first so genuine support is visible on top.
        order=np.argsort(result['inliers'],kind='stable')
        for i in order:
            x,y=src[i];u,v=dst[i];u+=1320;y+=90;v+=90;color='#00aa70' if result['inliers'][i] else '#ed4a2c'
            draw.line((x,y,u,v),fill=color,width=2)
            for xx,yy in [(x,y),(u,v)]:draw.ellipse((xx-4,yy-4,xx+4,yy+4),outline=color,width=2)
        image.thumbnail((2000,1400),Image.Resampling.LANCZOS);image.save(ROOT/f'figures/{pair}_inliers.jpg',quality=96)
        records.append(rec);print(pair,rec['input_count'],'inputs',rec['inlier_count'],'inliers','RMSE',round(rec['rmse_inlier_error_working_px'],3),'converged',numeric['refit_converged'],flush=True)
    assert all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    (ROOT/'automatic_homographies.json').write_text(json.dumps({'protocol':'B3 ratio<.7 plus mutual-nearest; symmetric RMS <=3 working px; four-point RANSAC; no manual H consulted','pairs':records,'input_sha256':hashes},indent=2,allow_nan=False),encoding='utf-8')
    print('Automatic models frozen. No A2 manual matrix file has been read.',flush=True)

if __name__=='__main__':main()
