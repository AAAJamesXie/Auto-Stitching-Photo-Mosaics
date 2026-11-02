"""Handwritten four-point RANSAC, using the unchanged A2 linear solver code."""
import itertools,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'A2'))
from homography import computeH

def degenerate_sample(points):
    points=np.asarray(points,float)
    if points.shape!=(4,2) or not np.isfinite(points).all() or len(np.unique(points,axis=0))<4:return True
    span=np.ptp(points,axis=0).max()
    if span<=0:return True
    p=(points-points.mean(axis=0))/span
    for a,b,c in itertools.combinations(range(4),3):
        u=p[b]-p[a];v=p[c]-p[a]
        if abs(u[0]*v[1]-u[1]*v[0])<1e-4:return True
    return False

def _project(points,H):
    q=np.c_[points,np.ones(len(points))]@H.T
    magnitude=np.abs(np.c_[points,np.ones(len(points))])@np.abs(H[2])
    safe=np.abs(q[:,2])>1e-12*np.maximum(magnitude,np.finfo(float).tiny)
    return np.divide(q[:,:2],q[:,2,None],out=np.full((len(points),2),np.nan),where=safe[:,None])

def symmetric_errors(source,target,H):
    H=np.asarray(H,float)
    if H.shape!=(3,3) or not np.isfinite(H).all() or not np.any(H):return np.full(len(source),np.inf)
    H=H/np.abs(H).max()
    try:inverse=np.linalg.inv(H)
    except np.linalg.LinAlgError:return np.full(len(source),np.inf)
    a=_project(source,H);b=_project(target,inverse)
    error=np.sqrt((np.sum((a-target)**2,axis=1)+np.sum((b-source)**2,axis=1))/2)
    return np.where(np.isfinite(error),error,np.inf)

def working_h_to_original(H,C_source,C_target):
    original=np.linalg.solve(C_target,np.asarray(H)@np.asarray(C_source))
    return original/original[2,2]

def ransac_homography(source,target,threshold=3.,seed=1337,max_iterations=5000,min_iterations=100,confidence=.999,min_inliers=8):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.ndim!=2 or source.shape[1]!=2 or target.shape!=source.shape or len(source)<max(4,min_inliers):
        raise ValueError('Enough paired Nx2 coordinates for the required consensus are needed.')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Finite correspondences required.')
    if not np.isfinite(threshold) or threshold<=0 or not 0<confidence<1 or not 4<=min_inliers<=len(source):raise ValueError('Invalid RANSAC threshold/confidence/consensus.')
    if not isinstance(max_iterations,int) or not isinstance(min_iterations,int) or not 1<=min_iterations<=max_iterations:raise ValueError('Invalid iteration limits.')
    rng=np.random.default_rng(seed);best=None;best_score=(-1,-np.inf);best_sample=None
    draws=0;hypotheses=0;degenerate=0;failed=0;required=max_iterations;updates=[]
    while draws<max_iterations and hypotheses<required:
        sample=rng.choice(len(source),4,replace=False);draws+=1
        if degenerate_sample(source[sample]) or degenerate_sample(target[sample]):degenerate+=1;continue
        try:H=computeH(source[sample],target[sample])
        except (ValueError,np.linalg.LinAlgError):failed+=1;continue
        hypotheses+=1;errors=symmetric_errors(source,target,H);mask=errors<=threshold
        score=(int(mask.sum()),-float(np.minimum(errors**2,threshold**2).sum()))
        if score>best_score:
            best_score=score;best=mask;best_sample=sample.copy()
            fraction=score[0]/len(source);success=fraction**4
            nominal=1 if success>=1 else math.ceil(math.log1p(-confidence)/math.log1p(-success)) if success>0 else max_iterations
            required=min(max_iterations,max(min_iterations,nominal))
            updates.append({'draw':draws,'hypotheses':hypotheses,'inliers':score[0],'required_hypotheses':required})
    if best is None or best.sum()<min_inliers:raise ValueError('No sufficient nondegenerate consensus found; no transform returned.')
    initial=best.copy();current=best.copy();history=[];converged=False
    for iteration in range(20):
        fitted=current.copy();H=computeH(source[fitted],target[fitted]);errors=symmetric_errors(source,target,H)
        current=errors<=threshold
        history.append({'iteration':iteration+1,'fitted_count':int(fitted.sum()),'classified_count':int(current.sum())})
        if current.sum()<min_inliers:raise ValueError('Consensus fell below minimum during least-squares refinement.')
        if np.array_equal(current,fitted):converged=True;break
    return {'H':H,'errors':errors,'inliers':current,'fitted_mask':fitted,'initial_consensus':initial,'best_sample':best_sample,
            'seed':seed,'threshold':threshold,'draws':draws,'hypotheses':hypotheses,'degenerate_draws':degenerate,'solver_failures':failed,
            'required_hypotheses':required,'confidence':confidence,'max_iterations':max_iterations,'min_iterations':min_iterations,
            'updates':updates,'refit_history':history,'refit_converged':converged,'hit_draw_limit':draws>=max_iterations}
