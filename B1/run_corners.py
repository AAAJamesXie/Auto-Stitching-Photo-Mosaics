"""Generate real-image B1 artifacts; no descriptors, matches or panorama."""
import hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from corners import harris_response,detect_candidates,anms,working_to_original
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
NAMES=['8443','8444','8445','8446','8447','8448','8450']
PARAMS={'working_size':[1200,1600],'grayscale':'0.299R+0.587G+0.114B on RGB/255','derivative_sigma':1.,'integration_sigma':1.5,'harris_k':.04,'response':'det(M)-k*trace(M)^2','threshold_rel':.001,'local_max_window':3,'border':24,'anms_robust':.9,'requested_count':500,'anms_block_size':128}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def spatial_metrics(xy,size):
    if len(xy)<2:return {'median_nearest_neighbor_px':None,'occupied_grid_cells':0,'grid_cells':48}
    d=np.sum((xy[:,None,:]-xy[None,:,:])**2,axis=2);np.fill_diagonal(d,np.inf)
    cell=np.floor(xy/np.array(size)*[6,8]).astype(int)
    return {'median_nearest_neighbor_px':float(np.median(np.sqrt(d.min(axis=1)))),'occupied_grid_cells':int(len(np.unique(cell,axis=0))),'grid_cells':48}

def overlay(image,xy,color):
    im=image.copy();draw=ImageDraw.Draw(im)
    for x,y in xy:draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
    return im

def main():
    for name in ['working','points','figures','inspection']:(ROOT/name).mkdir(exist_ok=True)
    tracked=list((PROJECT/'A1/images').glob('*.jpg'))+[PROJECT/'A2/homographies.json',PROJECT/'A3/warp.py',PROJECT/'A4/blend.py',PROJECT/'A4/results/panorama.png',PROJECT/'A4/panorama_measurements.json']
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in tracked};records=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    for name in NAMES:
        start=time.perf_counter()
        with Image.open(PROJECT/f'A1/images/IMG_{name}.jpg') as im:
            original_size=im.size;working=im.convert('RGB').resize(tuple(PARAMS['working_size']),Image.Resampling.LANCZOS)
        working.save(ROOT/f'working/IMG_{name}.png')
        rgb=np.array(working,dtype=np.float64)/255
        gray=rgb@np.array([.299,.587,.114])
        response=harris_response(gray)
        xy,strength=detect_candidates(response)
        selected,radii=anms(xy,strength)
        strongest=np.lexsort((xy[:,0],xy[:,1],-strength))[:len(selected)]
        original=working_to_original(xy,original_size,working.size)
        scale=np.array(working.size)/original_size
        H=np.array([[scale[0],0,(scale[0]-1)/2],[0,scale[1],(scale[1]-1)/2],[0,0,1.]])
        # Preserve all candidate scores/radii and selection indices for future B2.
        np.savez_compressed(ROOT/f'points/IMG_{name}.npz',xy_working=xy,xy_original=original,response=strength,radius=radii,anms_indices=selected,strongest_indices=strongest,H_original_to_working=H,original_size=original_size,working_size=working.size)
        points=[{'candidate_index':int(i),'working_xy':xy[i].tolist(),'original_xy':original[i].tolist(),'response':float(strength[i]),'suppression_radius':float(radii[i]) if np.isfinite(radii[i]) else None,'radius_unbounded':bool(np.isinf(radii[i]))} for i in selected]
        (ROOT/f'points/IMG_{name}.json').write_text(json.dumps({'source':f'IMG_{name}.jpg','working_image':f'../working/IMG_{name}.png','coordinate_rule':'original=(working+.5)*original_size/working_size-.5','H_original_to_working':H.tolist(),'selected_points':points},indent=2,allow_nan=False),encoding='utf-8')
        kinds=[('candidates',xy,'#ff5533',f'Harris candidates / {len(xy):,}'),('strongest',xy[strongest],'#ff5533',f'Strongest / {len(strongest)}'),('anms',xy[selected],'#00e5ff',f'ANMS / {len(selected)}')]
        comparison=Image.new('RGB',(1800,880),'#ece8e6');draw=ImageDraw.Draw(comparison)
        for j,(key,pts,color,label) in enumerate(kinds):
            annotated=overlay(working,pts,color);annotated.save(ROOT/f'figures/IMG_{name}_{key}.jpg',quality=96)
            annotated.thumbnail((570,760),Image.Resampling.LANCZOS);comparison.paste(annotated,(j*600+15,90));draw.text((j*600+15,20),label,font=font,fill='black')
        comparison.save(ROOT/f'figures/IMG_{name}_comparison.jpg',quality=96)
        np.testing.assert_allclose((original+.5)*scale-.5,xy,atol=1e-10)
        assert len(np.unique(selected))==len(selected) and len(selected)==min(500,len(xy))
        assert np.isfinite(response).all() and (strength>0).all()
        for i in np.linspace(0,max(0,len(xy)-1),min(20,len(xy))).astype(int):
            stronger=strength[i]<.9*strength
            expected=np.linalg.norm(xy[stronger]-xy[i],axis=1).min() if stronger.any() else np.inf
            assert np.isclose(radii[i],expected) or (np.isinf(radii[i]) and np.isinf(expected))
        rec={'image':f'IMG_{name}.jpg','original_size':list(original_size),'working_size':list(working.size),'candidate_count':len(xy),'selected_count':len(selected),'unbounded_radius_count':int(np.isinf(radii).sum()),'absolute_response_threshold':float(max(0,response.max()*.001)),'response_max':float(response.max()),'strongest_metrics':spatial_metrics(xy[strongest],working.size),'anms_metrics':spatial_metrics(xy[selected],working.size),'selected_finite_radius_min':float(np.min(radii[selected][np.isfinite(radii[selected])])) if np.isfinite(radii[selected]).any() else None,'H_original_to_working':H.tolist(),'points_npz':f'points/IMG_{name}.npz','points_json':f'points/IMG_{name}.json','seconds_including_io':time.perf_counter()-start}
        records.append(rec);print(json.dumps(rec),flush=True)
    assert all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    (ROOT/'measurements.json').write_text(json.dumps({'parameters':PARAMS,'records':records,'input_sha256':hashes,'software':{'numpy':np.__version__,'opencv':cv2.__version__},'reference':'https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf'},indent=2,allow_nan=False),encoding='utf-8')
    print('Seven B1 datasets generated; original inputs unchanged.',flush=True)

if __name__=='__main__':main()
