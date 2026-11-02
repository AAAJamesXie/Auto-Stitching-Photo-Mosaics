"""Compose adjacent manual homographies into a shared scaled reference frame."""
import numpy as np

def compose_transforms(names,reference_index,pairs,scale=.6):
    if not 0<=reference_index<len(names) or not 0<scale<=1:
        raise ValueError('Valid reference and scale in (0,1] required.')
    transforms={reference_index:np.eye(3)}
    for i in range(reference_index-1,-1,-1):
        transforms[i]=transforms[i+1]@np.asarray(pairs[names[i]+'_'+names[i+1]])
    for i in range(reference_index+1,len(names)):
        transforms[i]=transforms[i-1]@np.linalg.inv(pairs[names[i-1]+'_'+names[i]])
    S=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]])
    return [S@transforms[i] for i in range(len(names))]
