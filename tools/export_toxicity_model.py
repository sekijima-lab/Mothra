"""OFFLINE converter: run only in the trusted historical sklearn environment."""
import hashlib,sys
from pathlib import Path
import joblib,numpy as np
p=Path(sys.argv[1])
expected='792cce50db528a20c9769842a0efac26aa9a77e5e04ba76819ee500a78d29566'
if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
    raise ValueError('Refusing to unpickle a file other than the trusted bundled historical model')
model=joblib.load(p)
assert model.__class__.__name__=='ExtraTreesClassifier'
assert model.n_features_in_==1024 and np.array_equal(model.classes_,[0,1])
trees=[m.tree_ for m in model.estimators_]
values=[]
for tree in trees:
    v=tree.value[:,0,:].copy();v/=v.sum(1,keepdims=True);values.append(v)
np.savez_compressed(sys.argv[2],format_version=np.array(1),n_features=np.array(1024),classes=model.classes_,
    tree_offsets=np.r_[0,np.cumsum([t.node_count for t in trees])],
    children_left=np.concatenate([t.children_left for t in trees]),children_right=np.concatenate([t.children_right for t in trees]),
    feature=np.concatenate([t.feature for t in trees]),threshold=np.concatenate([t.threshold for t in trees]),probabilities=np.concatenate(values),
    source_sha256=np.array(hashlib.sha256(p.read_bytes()).hexdigest()))
print(len(trees),'trees',sum(t.node_count for t in trees),'nodes')
