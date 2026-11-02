"""Exact descriptor nearest neighbors and strict Lowe ratio test with NumPy."""
import numpy as np

def match_descriptors(source,target,source_valid=None,target_valid=None,ratio_threshold=.7,block_size=64):
    source=np.asarray(source,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1] or source.shape[1]<1:
        raise ValueError('Descriptor arrays must have matching nonzero dimensions.')
    sv=np.ones(len(source),bool) if source_valid is None else np.asarray(source_valid,dtype=bool)
    tv=np.ones(len(target),bool) if target_valid is None else np.asarray(target_valid,dtype=bool)
    if sv.shape!=(len(source),) or tv.shape!=(len(target),):raise ValueError('Validity flags must match row counts.')
    if not np.isfinite(source[sv]).all() or not np.isfinite(target[tv]).all():raise ValueError('Valid descriptors must be finite.')
    if not 0<ratio_threshold<1 or not isinstance(block_size,int) or block_size<1:raise ValueError('Threshold in (0,1) and positive integer block size required.')
    result={key:np.full(len(source),-1,dtype=np.int64) for key in ['nearest_target_rows','second_target_rows']}
    result.update({key:np.full(len(source),np.nan) for key in ['d1','d2','ratio','ratio_squared']})
    result.update({key:np.zeros(len(source),bool) for key in ['ratio_pass','mutual_nearest','ratio_mutual']})
    result['query_valid']=sv.copy()
    q=np.flatnonzero(sv);t=np.flatnonzero(tv)
    if not len(q) or len(t)<2:return result
    distances=np.empty((len(q),len(t)),float)
    for start in range(0,len(q),block_size):
        stop=min(start+block_size,len(q));difference=source[q[start:stop],None,:]-target[t][None,:,:]
        distances[start:stop]=np.sum(difference*difference,axis=2)
    order=np.argsort(distances,axis=1,kind='stable')[:,:2]
    d1sq=distances[np.arange(len(q)),order[:,0]];d2sq=distances[np.arange(len(q)),order[:,1]]
    d1=np.sqrt(d1sq);d2=np.sqrt(d2sq)
    ratio=np.divide(d1,d2,out=np.full(len(q),np.nan),where=d2>0)
    ratio_sq=np.divide(d1sq,d2sq,out=np.full(len(q),np.nan),where=d2sq>0)
    mutual=np.argmin(distances,axis=0)[order[:,0]]==np.arange(len(q))
    result['nearest_target_rows'][q]=t[order[:,0]];result['second_target_rows'][q]=t[order[:,1]]
    for key,values in [('d1',d1),('d2',d2),('ratio',ratio),('ratio_squared',ratio_sq),('ratio_pass',ratio<ratio_threshold),('mutual_nearest',mutual)]:result[key][q]=values
    result['ratio_mutual']=result['ratio_pass']&result['mutual_nearest']
    return result
