"""One full-resolution two-view Part A mosaic; no feature detection or RANSAC."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'A3'))
from warp import warpImageBilinear,output_canvas
from blend import feather_weights,blend_layers

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rgb8(a):return np.clip(np.rint(a),0,255).astype(np.uint8)
def save_rgba(a,mask,path):
    rgba=np.empty((*mask.shape,4),np.uint8);rgba[:,:,:3]=rgb8(a);rgba[:,:,3]=mask.astype(np.uint8)*255
    Image.fromarray(rgba).save(path)
def plate(images,labels,path,height=760):
    tiles=[]
    for im in images:
        im=im.convert('RGB');im.thumbnail((830,height),Image.Resampling.LANCZOS);tiles.append(im)
    out=Image.new('RGB',(1800,height+90),'#ece8e6');d=ImageDraw.Draw(out)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
    for i,(im,label) in enumerate(zip(tiles,labels)):
        d.text((i*900+24,16),label,font=font,fill='black');out.paste(im,(i*900+(900-im.width)//2,70))
    out.save(path,quality=95)

def main():
    for folder in ['results','figures','inspection']:(ROOT/folder).mkdir(exist_ok=True)
    inputs=[PROJECT/'A1/images'/f'IMG_{n}.jpg' for n in [8451,8452]]+[PROJECT/'A2/homographies.json',PROJECT/'A2/homography.py',PROJECT/'A3/warp.py']
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in inputs}
    pair=next(p for p in json.loads(inputs[2].read_text(encoding='utf-8'))['pairs'] if p['id']=='8451_8452')
    H=np.array(pair['H_source_to_target']); source=np.array(Image.open(inputs[0]).convert('RGB'));reference=np.array(Image.open(inputs[1]).convert('RGB'))
    x,y,w,h=output_canvas(source.shape,H);rh,rw=reference.shape[:2]
    x0=min(x,0);y0=min(y,0);x1=max(x+w-1,rw-1);y1=max(y+h-1,rh-1)
    canvas=(x0,y0,x1-x0+1,y1-y0+1);_,_,cw,ch=canvas
    start=time.perf_counter()
    warped=warpImageBilinear(source,H,canvas=canvas)
    weight_result=warpImageBilinear(feather_weights(source.shape),H,canvas=canvas)
    wa=weight_result.image;wa[~warped.alpha]=0; del weight_result
    wb=np.zeros((ch,cw),np.float32);oy,ox=-y0,-x0
    wb[oy:oy+rh,ox:ox+rw]=feather_weights(reference.shape)
    ma=warped.alpha;mb=wb>0;union=ma|mb;overlap=ma&mb
    # Baseline: reference overwrites source wherever reference is valid.
    baseline=rgb8(warped.image);baseline[oy:oy+rh,ox:ox+rw]=reference
    save_rgba(baseline,union,ROOT/'results/overwrite.png')
    box=(1700,1250,1940,1490) # canvas pixels, same architectural ROI for both outputs
    def layers():
        yield warped.image,wa
        pixels=np.zeros((ch,cw,3),np.uint8);pixels[oy:oy+rh,ox:ox+rw]=reference
        yield pixels,wb
    blended,valid=blend_layers(layers());assert np.array_equal(valid,union)
    save_rgba(blended,valid,ROOT/'results/mosaic.png')
    for name,mask in [('union',union),('overlap',overlap),('source',ma),('reference',mb)]:Image.fromarray(mask.astype(np.uint8)*255).save(ROOT/f'results/{name}_mask.png')
    total=wa+wb; normalized=np.divide(wa,total,out=np.zeros_like(wa),where=total>0)
    Image.fromarray(np.rint(normalized*65535).astype(np.uint16)).save(ROOT/'results/source_weight.png')
    # RGB color map: orange=source, blue=reference, gray=outside union.
    color=np.empty((ch,cw,3),np.uint8)
    for k,(a,b) in enumerate(zip([245,100,30],[30,120,230])):color[:,:,k]=np.rint(normalized*a+(1-normalized)*b).astype(np.uint8)
    color[~union]=[236,232,230]
    im=Image.fromarray(color);im.thumbnail((1600,1600));im.save(ROOT/'figures/weights.jpg',quality=95)
    def display(a):
        im=Image.fromarray(rgb8(a));im.paste((236,232,230),(0,0),Image.fromarray((~union).astype(np.uint8)*255));return im
    plate([display(baseline),display(blended)],['Reference overwrites source','Feathered weighted average'],ROOT/'figures/comparison.jpg')
    final=display(blended);final.thumbnail((1700,1700));final.save(ROOT/'figures/mosaic.jpg',quality=96)
    b0=Image.fromarray(baseline).crop(box).resize((720,720),Image.Resampling.NEAREST)
    b1=Image.fromarray(rgb8(blended)).crop(box).resize((720,720),Image.Resampling.NEAREST)
    plate([b0,b1],['Overwrite / same 240 x 240 crop','Feather / pixels replicated 3x'],ROOT/'figures/details.jpg',height=740)
    seam_box=(1063,1800,1303,2040)
    seam=[Image.fromarray(rgb8(a)).crop(seam_box).resize((720,720),Image.Resampling.NEAREST) for a in [baseline,blended]]
    plate(seam,['Overwrite / reference left edge','Feather / same boundary crop'],ROOT/'figures/seam.jpg',height=740)
    # Direct scalar reconstruction at genuine overlap and sole-support samples.
    probes=[]
    for name,mask in [('overlap',overlap),('source_only',ma&~mb),('reference_only',mb&~ma)]:
        yy,xx=np.nonzero(mask)
        for j in [len(xx)//4,len(xx)//2,3*len(xx)//4]:
            u,v=int(xx[j]),int(yy[j]);b=reference[v-oy,u-ox].astype(float) if mb[v,u] else np.zeros(3)
            expected=(warped.image[v,u]*float(wa[v,u])+b*float(wb[v,u]))/float(total[v,u])
            np.testing.assert_allclose(blended[v,u],expected,atol=3e-5,rtol=1e-6)
            probes.append({'region':name,'canvas_xy':[u,v],'expected_rgb':expected.tolist(),'actual_rgb':blended[v,u].tolist()})
    single=union&~overlap
    np.testing.assert_array_equal(rgb8(blended)[single],baseline[single])
    assert all(sha(PROJECT/p)==digest for p,digest in hashes.items())
    metadata={'clarification':{'part_A_mosaics':1,'part_A_min_images':2,'part_B_mosaics':1,'part_B_min_images':3},'images':[p.name for p in inputs[:2]],'reference':'IMG_8452.jpg','H_source_to_reference':H.tolist(),'H_source_to_canvas':warped.H_source_to_canvas.tolist(),'canvas':list(canvas),'valid_pixels':int(union.sum()),'overlap_pixels':int(overlap.sum()),'source_only_pixels':int((ma&~mb).sum()),'reference_only_pixels':int((mb&~ma).sum()),'detail_canvas_box':list(box),'A2_check_rmse_px':pair['check_rmse_px'],'weight_rule':'min(x+1,w-x,y+1,h-y), divided by source maximum; bilinear warp; invalid=0; normalize sum at destination','overlap_mean_abs_rgb_difference':float(np.abs(warped.image[oy:oy+rh,ox:ox+rw][overlap[oy:oy+rh,ox:ox+rw]]-reference[overlap[oy:oy+rh,ox:ox+rw]]).mean()),'input_sha256':hashes,'verification_probes':probes,'elapsed_seconds_including_io':time.perf_counter()-start}
    (ROOT/'measurements.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['canvas','valid_pixels','overlap_pixels','elapsed_seconds_including_io']}))

if __name__=='__main__':main()
