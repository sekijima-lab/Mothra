import os,sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import numpy as np,tensorflow as tf,keras
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.random.set_seed(1729);np.random.seed(1729)
sys.path.insert(0,str(root/'ligand_design'));os.chdir(str(root))
from make_smile import zinc_data_with_bracket_original,zinc_processed_with_bracket
from load_model import loaded_model,prepare_data
if not hasattr(keras.utils,'pad_sequences'):
    keras.utils.pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
from keras.utils import pad_sequences
from rdkit import Chem,rdBase
from rdkit.Chem import AllChem,QED
rdBase.DisableLog('rdApp.*')
old_compatible='--no-old-compatible' not in sys.argv
model=loaded_model(str(root/'model3')+'/',old_compatible=old_compatible) if int(keras.__version__.split('.')[0])>=3 else loaded_model(str(root/'model3')+'/')
smiles=zinc_data_with_bracket_original()
vocab,tokenized=zinc_processed_with_bracket(smiles)
assert len(vocab)==64
xs,ys=prepare_data(vocab,tokenized[:256])
X=pad_sequences(xs,maxlen=81,dtype='int32',padding='post',truncating='pre',value=0)
y=pad_sequences(ys,maxlen=81,dtype='int32',padding='post',truncating='pre',value=0)
pred=model(X,training=False).numpy()
np.savez_compressed(out/'prediction.npz',inputs=X,targets=y,probabilities=pred)
np.savez_compressed(out/'initial-weights.npz',**{'weight_%02d'%i:v.numpy() for i,v in enumerate(model.weights)})
if int(keras.__version__.split('.')[0]) >= 3:
    sys.path.insert(0,str(root/'train_RNN'))
    from optimizers import DeduplicatingAdam
    optimizer=(DeduplicatingAdam if old_compatible else keras.optimizers.Adam)(learning_rate=0.0001)
else:
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
with tf.GradientTape() as tape:
    probabilities=model(X[:16],training=False)
    labels=tf.one_hot(y[:16],64)
    loss=keras.losses.CategoricalCrossentropy()(labels,probabilities)
grad=tape.gradient(loss,model.trainable_variables)
optimizer.apply_gradients(zip(grad,model.trainable_variables))
np.savez_compressed(out/'training.npz',**{'gradient_%02d'%i:tf.convert_to_tensor(g).numpy() for i,g in enumerate(grad)},**{'weight_%02d'%i:v.numpy() for i,v in enumerate(model.weights)})
from sascorer import calculateScore
toxic_smiles=smiles[:2048]
fingerprints=[];rewards=[]
for s in toxic_smiles:
 m=Chem.MolFromSmiles(s);assert m is not None
 h=Chem.AddHs(m);bits=AllChem.GetMorganFingerprintAsBitVect(h,2,nBits=1024).ToBitString()
 fingerprints.append(np.array(list(bits),dtype=float))
 rewards.append([QED.default(m),calculateScore(m)])
fingerprints=np.asarray(fingerprints)
if int(keras.__version__.split('.')[0])<3:
 import joblib
 predictor=joblib.load(root/'ligand_design/etoxpred_best_model.joblib')
else:
 from toxicity import ToxicityPredictor
 predictor=ToxicityPredictor()
tox=predictor.predict_proba(fingerprints)
np.savez_compressed(out/'toxicity.npz',fingerprints=fingerprints,probabilities=tox,rewards=np.array(rewards))
summary={'old_compatible':old_compatible,'python':sys.version,'tensorflow':tf.__version__,'keras':keras.__version__,'numpy':np.__version__,'vocabulary':vocab,'loss':float(loss.numpy()),'prediction_rows':len(X),'toxicity_rows':len(tox),'accepted':int((tox[:,1]<0.7).sum()),'source_hashes':{s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in ['model3/model.json','model3/model.h5','data/250k_rndm_zinc_drugs_clean.smi']}}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
