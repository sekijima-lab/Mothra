import sys,json
from pathlib import Path
import numpy as np
old=sys.argv[1]=='legacy';out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
if old:
 import pygmo
 def hv(points):return pygmo.hypervolume(points).compute([0,0,0])
else:
 import moocore
 def hv(points):return float(moocore.hypervolume(points,ref=[0,0,0]))
fixtures=[np.array([[-1,-2,-3]]),np.array([[-1,-2,-3],[-2,-1,-3]]),np.array([[-1,-2,-3],[-1,-2,-3],[-1e-17,-1e-17,-1e-17]])]
rng=np.random.RandomState(1729)
fixtures.extend([-rng.rand(n,3)*[10,1,1] for n in [2,5,10,50,100,1000] for i in range(10)])
values=[];rankings=[]
for points in fixtures:
 values.append(hv(points));candidates=-rng.rand(12,3)*[10,1,1]
 scores=[hv(np.vstack([points,p])) for p in candidates];rankings.append(np.argsort(scores,kind='stable').tolist())
out.write_text(json.dumps({'values':values,'candidate_rankings':rankings},indent=2));print(len(values),'fixtures')
