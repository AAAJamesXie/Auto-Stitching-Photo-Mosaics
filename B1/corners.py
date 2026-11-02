"""Single-scale classical Harris response and exact blocked Brown ANMS.

OpenCV supplies only Gaussian smoothing and neighborhood maximum filtering;
no feature detector, descriptor, matcher or high-level stitching API is used.
"""
import cv2
import numpy as np

def harris_response(gray,k=.04,sigma_d=1.,sigma_i=1.5):
    gray=np.asarray(gray,dtype=np.float64)
    if gray.ndim!=2 or min(gray.shape)<3 or not np.isfinite(gray).all():
        raise ValueError('Finite 2D grayscale image with dimensions >=3 required.')
    if not 0<k<.25 or sigma_d<=0 or sigma_i<=0:
        raise ValueError('Positive smoothing scales and k in (0,.25) required.')
    smooth=cv2.GaussianBlur(gray,(0,0),sigma_d,borderType=cv2.BORDER_REFLECT_101)
    gy,gx=np.gradient(smooth)
    a=cv2.GaussianBlur(gx*gx,(0,0),sigma_i,borderType=cv2.BORDER_REFLECT_101)
    b=cv2.GaussianBlur(gx*gy,(0,0),sigma_i,borderType=cv2.BORDER_REFLECT_101)
    c=cv2.GaussianBlur(gy*gy,(0,0),sigma_i,borderType=cv2.BORDER_REFLECT_101)
    return a*c-b*b-k*(a+c)**2

def detect_candidates(response,threshold_rel=.001,border=24):
    response=np.asarray(response,dtype=np.float64)
    if response.ndim!=2 or not response.size or not np.isfinite(response).all():
        raise ValueError('Finite nonempty response map required.')
    if not 0<=threshold_rel<1 or not isinstance(border,int) or border<1:
        raise ValueError('Threshold in [0,1) and integer border>=1 required.')
    local=cv2.dilate(response,np.ones((3,3),np.uint8))
    valid=(response>max(0.,float(response.max())*threshold_rel))&(response==local)
    valid[:border]=False;valid[-border:]=False;valid[:,:border]=False;valid[:,-border:]=False
    y,x=np.nonzero(valid)
    # Each pixel has a unique coordinate. Exact plateau ties remain separate
    # candidates and are resolved deterministically at the selection stage.
    return np.c_[x,y].astype(np.float64),response[y,x]

def anms(xy,strength,count=500,robust=.9,block_size=128):
    """r_i=min distance to j with strength_i < robust*strength_j.

    Compute radii against every candidate, not a capped strongest subset.
    No eligible neighbor -> infinity; rank by decreasing radius, then strength,
    then increasing y,x. Return selected candidate indices and all radii.
    """
    xy=np.asarray(xy,dtype=np.float64);strength=np.asarray(strength,dtype=np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or strength.shape!=(len(xy),):
        raise ValueError('Nx2 coordinates and N strengths required.')
    if not np.isfinite(xy).all() or not np.isfinite(strength).all() or (strength<=0).any():
        raise ValueError('Finite coordinates and positive finite strengths required.')
    if not 0<robust<=1 or not isinstance(count,int) or count<0 or not isinstance(block_size,int) or block_size<1:
        raise ValueError('Valid robust factor, nonnegative integer count and positive block size required.')
    radii=np.full(len(xy),np.inf)
    for start in range(0,len(xy),block_size):
        stop=min(start+block_size,len(xy))
        dx=xy[start:stop,0,None]-xy[None,:,0];dy=xy[start:stop,1,None]-xy[None,:,1]
        distances=dx*dx+dy*dy
        eligible=strength[start:stop,None]<robust*strength[None,:]
        distances[~eligible]=np.inf
        radii[start:stop]=np.sqrt(distances.min(axis=1))
    order=np.lexsort((xy[:,0],xy[:,1],-strength,-radii))
    return order[:count],radii

def working_to_original(xy,original_size,working_size):
    return (np.asarray(xy,dtype=np.float64)+.5)*np.asarray(original_size)/np.asarray(working_size)-.5
