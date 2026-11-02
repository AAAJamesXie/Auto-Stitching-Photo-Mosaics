"""Mask-aware feathering with handwritten NumPy weighted accumulation."""
import numpy as np


def feather_weights(shape):
    """Positive distance to the outside edge, normalized to one at the center.

    Pixel centers at the border retain weight 1/max_distance, so sole-support
    border pixels never disappear. Invalid warped samples get zero separately.
    """
    h,w=shape[:2]
    if h<1 or w<1: raise ValueError('Nonempty source required.')
    x=np.minimum(np.arange(w)+1,w-np.arange(w))
    y=np.minimum(np.arange(h)+1,h-np.arange(h))
    weights=np.minimum(y[:,None],x[None,:]).astype(np.float32)
    return weights/weights.max()


def blend_layers(layers):
    """Consume (H x W x 3 RGB, H x W nonnegative weight) layers sequentially."""
    total=None; denominator=None
    for pixels,weight in layers:
        pixels=np.asarray(pixels);weight=np.asarray(weight,dtype=np.float32)
        if pixels.ndim!=3 or pixels.shape[2]!=3 or weight.shape!=pixels.shape[:2]:
            raise ValueError('Expected RGB pixels and matching 2D weights.')
        if not np.isfinite(pixels).all() or not np.isfinite(weight).all() or (weight<0).any():
            raise ValueError('Finite pixels and finite nonnegative weights required.')
        if total is None:
            total=np.zeros(pixels.shape,np.float32);denominator=np.zeros(weight.shape,np.float32)
        if pixels.shape!=total.shape: raise ValueError('Layers must share a canvas.')
        for row in range(0,len(weight),64):
            sl=slice(row,row+64)
            old=denominator[sl];new=old+weight[sl]
            ratio=np.divide(weight[sl],new,out=np.zeros_like(new),where=new>0)
            # Online weighted mean: single support is copied with ratio exactly 1,
            # avoiding an unnecessary multiply/divide that can change rounding.
            total[sl]+=ratio[...,None]*(pixels[sl]-total[sl])
            denominator[sl]=new
    if total is None: raise ValueError('At least one layer required.')
    valid=denominator>0
    return total,valid
