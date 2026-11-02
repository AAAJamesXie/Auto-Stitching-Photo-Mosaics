"""Axis-aligned 40x40 support -> smoothed8x8 -> normalized64D.

No feature/descriptor API. OpenCV supplies the ordinary Gaussian convolution;
all sampling, validity and descriptor arithmetic are explicit NumPy operations.
"""
import cv2
import numpy as np

PATCH_OFFSETS=np.arange(40,dtype=float)-19.5
SAMPLE_INDICES=np.arange(8)*5+2

def _sample40(gray,xy):
    xs=xy[:,0,None,None]+PATCH_OFFSETS[None,None,:]
    ys=xy[:,1,None,None]+PATCH_OFFSETS[None,:,None]
    x0=np.floor(xs).astype(int);y0=np.floor(ys).astype(int)
    x1=np.minimum(x0+1,gray.shape[1]-1);y1=np.minimum(y0+1,gray.shape[0]-1)
    dx=xs-x0;dy=ys-y0
    return ((1-dx)*(1-dy)*gray[y0,x0]+dx*(1-dy)*gray[y0,x1]
            +(1-dx)*dy*gray[y1,x0]+dx*dy*gray[y1,x1])

def extract_descriptors(gray,xy,std_threshold=1e-6):
    """Preserve input row identity, including invalid zero-vector rows.

    Input is finite floating grayscale (no clipping), coordinates are (x,y)
    working-image pixel centers. Samples are centered symmetrically about each
    point. Returns raw40, blurred40, sampled8, mean/std, in_bounds/valid and Nx64.
    """
    gray=np.asarray(gray,dtype=np.float64);xy=np.asarray(xy,dtype=np.float64)
    if gray.ndim!=2 or min(gray.shape)<2 or not np.isfinite(gray).all():
        raise ValueError('Finite nonempty 2D grayscale image required.')
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
        raise ValueError('Finite Nx2 corner coordinates required.')
    if not np.isfinite(std_threshold) or std_threshold<=0:
        raise ValueError('Positive finite standard-deviation threshold required.')
    inside=((xy[:,0]-19.5>=0)&(xy[:,0]+19.5<=gray.shape[1]-1)
            &(xy[:,1]-19.5>=0)&(xy[:,1]+19.5<=gray.shape[0]-1))
    raw=np.zeros((len(xy),40,40),float);blurred=np.zeros_like(raw)
    if inside.any():
        smooth=cv2.GaussianBlur(gray,(13,13),2.,borderType=cv2.BORDER_REFLECT_101)
        raw[inside]=_sample40(gray,xy[inside]);blurred[inside]=_sample40(smooth,xy[inside])
    sampled=blurred[:,SAMPLE_INDICES[:,None],SAMPLE_INDICES[None,:]]
    flat=sampled.reshape(len(xy),64);mean=flat.mean(axis=1);std=flat.std(axis=1,ddof=0)
    valid=inside&(std>std_threshold)
    descriptors=np.zeros_like(flat)
    descriptors[valid]=(flat[valid]-mean[valid,None])/std[valid,None]
    return {'raw40':raw,'blurred40':blurred,'sampled8':sampled,'mean':mean,'std':std,
            'in_bounds':inside,'valid':valid,'descriptors':descriptors}
