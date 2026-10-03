"""Compare primitive kernels on identical historical states (no accumulated error)."""
import os,sys,json
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np,tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(1)
p=Path('mothra-runtime-validation');layers=np.load(p/'legacy-layers.npz');weights=np.load(p/'legacy-arm/initial-weights.npz');values={}
for layer,kernel_idx,bias_idx,input_key,hidden_key in [(1,1,3,'layer_1','layer_2'),(2,4,6,'layer_3','layer_4')]:
 kernel=tf.constant(weights['weight_%02d'%kernel_idx]);rec=tf.constant(weights['weight_%02d'%(kernel_idx+1)]);bias=weights['weight_%02d'%bias_idx]
 for t in [0,1,2,3,19,35,68]:
  prefix='gru%d_t%d_'%(layer,t)
  x=tf.constant(layers[input_key][:,t]);h=tf.zeros((256,256)) if t==0 else tf.constant(layers[hidden_key][:,t-1])
  mx=tf.matmul(x,kernel);mi=tf.matmul(h,rec)
  values[prefix+'input_matmul']=mx.numpy();values[prefix+'recurrent_matmul']=mi.numpy()
  mx=tf.nn.bias_add(mx,bias[0]);mi=tf.nn.bias_add(mi,bias[1]);xz,xr,xh=tf.split(mx,3,1);iz,ir,ih=tf.split(mi,3,1)
  z=tf.sigmoid(xz+iz);r=tf.sigmoid(xr+ir);candidate=tf.tanh(xh+r*ih);state=z*h+(1-z)*candidate
  for name,tensor in [('z',z),('r',r),('candidate',candidate),('state',state)]:values[prefix+name]=tensor.numpy()
# Isolate activation kernels, independent of matmul differences.
grid=tf.constant(np.linspace(-20,20,100001,dtype=np.float32))
values['sigmoid_grid']=tf.sigmoid(grid).numpy();values['tanh_grid']=tf.tanh(grid).numpy()
np.savez_compressed(sys.argv[1],**values)
print(json.dumps({'tensorflow':tf.__version__,'operations':len(values)}))
