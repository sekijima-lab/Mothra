import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d

def rewardtoDocking(r: float)->float:
    return 10*r/(r-1)

dataDir = "6lu7/log336h_6lu7/data10/"
df = pd.read_csv(dataDir+'/present/scores.csv')
arr = df.values # points include non-pareto-dominate points

blist = json.load(open(dataDir+'/present/pareto.json','r'))
brr = np.array(blist['front'])
myVecFunc = np.vectorize(rewardtoDocking)
brr[:,0] = myVecFunc(brr[:,0])
myFig = plt.figure()
ax = myFig.add_subplot(111,projection='3d')
for br in brr:
    arr = np.delete(arr, np.where(arr[:,1] == br[1]), axis=0)
ax.scatter(arr[:,0],arr[:,1],arr[:,2],color='blue',alpha = 0.7, picker=True)
ax.scatter(brr[:,0],brr[:,1],brr[:,2],color='red',alpha = 1.0, picker=True)

ax.set_xlim3d(min(np.concatenate([arr[:,0], brr[:,0]])), max(np.concatenate([arr[:,0], brr[:,0]])))
ax.set_ylim3d(min(np.concatenate([arr[:,1], brr[:,1]])), max(np.concatenate([arr[:,1], brr[:,1]])))
ax.set_zlim3d(min(np.concatenate([arr[:,2], brr[:,2]])), max(np.concatenate([arr[:,2], brr[:,2]])))

ax.set_xlabel("Docking Score")
ax.set_ylabel("QED score")
ax.set_zlabel("Toxicity Probability")
plt.savefig(dataDir+"present/front.png")