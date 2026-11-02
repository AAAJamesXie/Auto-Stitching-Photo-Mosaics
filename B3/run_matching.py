"""Six adjacent image pairs; descriptor-only matching without geometry."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matching import match_descriptors
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
NAMES=['8443','8444','8445','8446','8447','8448','8450']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scalar(v):return float(v) if np.isfinite(v) else None
def load(n):
    with np.load(PROJECT/f'B2/data/IMG_{n}.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}

def make_overlay(a,b,sa,tb,rows,result,path):
    width,height=1200,1600;gap=120
    image=Image.new('RGB',(2*width+gap,height+90),'#ece8e6');image.paste(a,(0,90));image.paste(b,(width+gap,90))
    draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    draw.text((20,20),f'{path.stem[:9]} / {len(rows)} ratio matches displayed',font=font,fill='black')
    for i in rows:
        j=result['nearest_target_rows'][i];x,y=sa['xy_working'][i];u,v=tb['xy_working'][j];u+=width+gap;y+=90;v+=90
        color='#00d9ed' if result['mutual_nearest'][i] else '#ff6933'
        draw.line((x,y,u,v),fill=color,width=2)
        for xx,yy in [(x,y),(u,v)]:draw.ellipse((xx-5,yy-5,xx+5,yy+5),outline=color,width=3)
    image.thumbnail((2000,1500),Image.Resampling.LANCZOS);image.save(path,quality=96)

def normalized_tile(a):
    return Image.fromarray(np.rint(np.clip((a.reshape(8,8)+3)/6,0,1)*255).astype(np.uint8)).convert('RGB').resize((160,160),Image.Resampling.NEAREST)

def main():
    for name in ['data','figures','inspection']:(ROOT/name).mkdir(exist_ok=True)
    tracked=[PROJECT/'B2/measurements.json',PROJECT/'B2/descriptors.py',PROJECT/'B1/corners.py',PROJECT/'A2/homographies.json',PROJECT/'A4/results/panorama.png']
    for n in NAMES:tracked.extend([PROJECT/f'B2/data/IMG_{n}.npz',PROJECT/f'B1/working/IMG_{n}.png'])
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in tracked};records=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for n,m in zip(NAMES[:-1],NAMES[1:]):
        start=time.perf_counter();pair=f'{n}_{m}';s=load(n);t=load(m)
        r=match_descriptors(s['descriptors'],t['descriptors'],s['valid'],t['valid'])
        accepted=np.flatnonzero(r['ratio_pass']);ranked=accepted[np.argsort(r['ratio'][accepted],kind='stable')]
        shown=ranked[np.linspace(0,len(ranked)-1,min(60,len(ranked))).astype(int)] if len(ranked) else np.empty(0,int)
        nearest=r['nearest_target_rows'][accepted]
        arrays=dict(r)
        arrays.update(source_rows=np.arange(len(s['valid'])),match_source_rows=accepted,match_target_rows=nearest,
                      match_source_feature_ids=s['feature_ids'][accepted],match_target_feature_ids=t['feature_ids'][nearest],
                      match_source_candidate_indices=s['candidate_indices'][accepted],match_target_candidate_indices=t['candidate_indices'][nearest],
                      match_source_xy_working=s['xy_working'][accepted],match_target_xy_working=t['xy_working'][nearest],
                      match_source_xy_original=s['xy_original'][accepted],match_target_xy_original=t['xy_original'][nearest])
        np.savez_compressed(ROOT/f'data/{pair}.npz',**arrays)
        audit=[]
        for i in range(len(s['valid'])):
            j=int(r['nearest_target_rows'][i]);k=int(r['second_target_rows'][i])
            status='invalid_source' if not s['valid'][i] else ('insufficient_targets' if j<0 else ('zero_second_distance' if r['d2'][i]==0 else ('ratio_pass' if r['ratio_pass'][i] else 'ratio_reject')))
            audit.append({'source_b2_row':i,'source_feature_id':str(s['feature_ids'][i]),'source_candidate_index':int(s['candidate_indices'][i]),'source_working_xy':s['xy_working'][i].tolist(),'source_original_xy':s['xy_original'][i].tolist(),'nearest_target_b2_row':j,'second_target_b2_row':k,'nearest_target_feature_id':str(t['feature_ids'][j]) if j>=0 else None,'nearest_target_candidate_index':int(t['candidate_indices'][j]) if j>=0 else None,'nearest_target_working_xy':t['xy_working'][j].tolist() if j>=0 else None,'nearest_target_original_xy':t['xy_original'][j].tolist() if j>=0 else None,'d1':scalar(r['d1'][i]),'d2':scalar(r['d2'][i]),'euclidean_ratio':scalar(r['ratio'][i]),'squared_ratio':scalar(r['ratio_squared'][i]),'ratio_pass':bool(r['ratio_pass'][i]),'mutual_nearest':bool(r['mutual_nearest'][i]),'ratio_and_mutual':bool(r['ratio_mutual'][i]),'status':status})
        (ROOT/f'data/{pair}.json').write_text(json.dumps({'source':f'IMG_{n}.jpg','target':f'IMG_{m}.jpg','threshold':.7,'criterion':'strict Euclidean d1/d2<0.7','queries':audit},indent=2,allow_nan=False),encoding='utf-8')
        with Image.open(PROJECT/f'B1/working/IMG_{n}.png') as im:a=im.convert('RGB')
        with Image.open(PROJECT/f'B1/working/IMG_{m}.png') as im:b=im.convert('RGB')
        make_overlay(a,b,s,t,shown,r,ROOT/f'figures/{pair}_matches.jpg')
        make_overlay(a,b,s,t,accepted,r,ROOT/f'figures/{pair}_all_matches.jpg')
        # Example choice is deterministic and uses only descriptor ratio ranks.
        examples=[]
        if len(ranked):
            for rank,label in [(0,'lowest accepted ratio'),(len(ranked)//2,'median accepted ratio'),(len(ranked)-1,'highest accepted ratio')]:
                row=int(ranked[rank])
                if row not in [x['source_row'] for x in examples]:examples.append({'source_row':row,'selection':label})
        rejected=np.flatnonzero(np.isfinite(r['ratio'])&~r['ratio_pass'])
        if len(rejected):examples.append({'source_row':int(rejected[np.argmin(r['ratio'][rejected])]),'selection':'lowest rejected ratio'})
        sheet=Image.new('RGB',(1800,max(1,len(examples))*310),'#ece8e6');draw=ImageDraw.Draw(sheet)
        for row,e in enumerate(examples):
            i=e['source_row'];j=int(r['nearest_target_rows'][i]);k=int(r['second_target_rows'][i]);y=row*310
            draw.text((20,y+10),f'{e["selection"]} / ratio {r["ratio"][i]:.4f} / mutual {bool(r["mutual_nearest"][i])}',font=font,fill='black')
            for column,(im,ds,idx,label,distance) in enumerate([(a,s,i,'Source',None),(b,t,j,'Nearest',r['d1'][i]),(b,t,k,'Second',r['d2'][i])]):
                x=column*600+20;cx,cy=ds['xy_working'][idx]
                crop=im.crop((int(cx)-20,int(cy)-20,int(cx)+20,int(cy)+20)).resize((160,160),Image.Resampling.NEAREST)
                sheet.paste(crop,(x,y+90));sheet.paste(normalized_tile(ds['descriptors'][idx]),(x+190,y+90))
                draw.text((x,y+55),f'{label} B2 row {idx}'+(f' / d={distance:.4f}' if distance is not None else ''),font=small,fill='black')
                draw.text((x,y+260),'RGB context          normalized descriptor [−3,+3]',font=small,fill='black')
            e.update(nearest_row=j,second_row=k,ratio=float(r['ratio'][i]))
        sheet.save(ROOT/f'figures/{pair}_examples.jpg',quality=97)
        counts={str(v):int((r['ratio']<v).sum()) for v in [.6,.7,.8]}
        # Direct scalar recomputation on spread queries, without matrix identity tricks.
        for i in np.linspace(0,len(s['valid'])-1,20).astype(int):
            distance=np.array([np.linalg.norm(s['descriptors'][i]-v) if ok else np.inf for v,ok in zip(t['descriptors'],t['valid'])])
            order=np.argsort(distance,kind='stable')[:2]
            assert r['nearest_target_rows'][i]==order[0] and r['second_target_rows'][i]==order[1]
            np.testing.assert_allclose([r['d1'][i],r['d2'][i]],distance[order],atol=1e-12)
        np.testing.assert_allclose(r['ratio_squared'][np.isfinite(r['ratio'])],r['ratio'][np.isfinite(r['ratio'])]**2,atol=1e-12)
        records.append({'pair':pair,'source':f'IMG_{n}.jpg','target':f'IMG_{m}.jpg','valid_source':int(s['valid'].sum()),'valid_target':int(t['valid'].sum()),'ratio_matches':len(accepted),'mutual_ratio_matches':int(r['ratio_mutual'].sum()),'unique_ratio_targets':int(len(np.unique(nearest))),'threshold_counts':counts,'shown_source_rows':shown.tolist(),'example_queries':examples,'seconds_including_io':time.perf_counter()-start})
        print(json.dumps(records[-1]),flush=True)
    assert all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    (ROOT/'measurements.json').write_text(json.dumps({'parameters':{'threshold':.7,'ratio':'Euclidean d1/d2; squared-distance ratio threshold mathematically0.49','distance':'exact float64 direct sum of64 squared component differences, then sqrt','distance_block_size':64,'mutual':'reverse nearest neighbor on all valid source rows; annotate, do not replace ratio filter','tie_rule':'ascending original B2 row','overlay_selection':'up to60 evenly spaced ranks after sorting all ratio passes by ascending ratio','geometry':'none; no homographies or RANSAC used'},'records':records,'input_sha256':hashes},indent=2,allow_nan=False),encoding='utf-8')
    print('Six descriptor-only match datasets saved; upstream inputs unchanged.',flush=True)

if __name__=='__main__':main()
