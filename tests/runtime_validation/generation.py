import os,sys,json,tempfile
from pathlib import Path
summary_path=Path(sys.argv[3]).resolve()
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import tensorflow as tf,numpy as np,keras
tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(1)
if not hasattr(keras.utils,'pad_sequences'):keras.utils.pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
sys.path.insert(0,str(root/'ligand_design'));os.chdir(str(root))
from load_model import loaded_model
from add_node_type import expanded_node,chem_kn_simulation,check_node_type
from unittest.mock import patch
vocab=json.loads(summary_path.read_text())['vocabulary'];model=loaded_model(str(root/'model3')+'/')
generation=[]
for prefix in [['&','C'],['&','C','C'],['&','c','1']]:
 for seed in [0,1,2]:
  np.random.seed(seed)
  nodes=expanded_node(model,prefix,vocab)
  selected=[vocab[i] for i in nodes[:3]]
  sequences=chem_kn_simulation(model,prefix,vocab,selected)
  generation.append({'prefix':prefix,'seed':seed,'nodes':nodes,'sequences':sequences})

smiles=['C','CC','CCC','CCCC','CCO','CCCO','CC(C)O','CCN','CC(=O)O','CC(=O)N','COC','CCOC','c1ccccc1','Oc1ccccc1','Nc1ccccc1','Cc1ccccc1','c1ccncc1','c1ccoc1','C1CCCCC1','C1CCNCC1','CCCl','CCBr','CC(F)F','CS','CCS','CC#N','O=C=O','CC(=O)Oc1ccccc1C(=O)O','Cn1c(=O)c2c(ncn2C)n(C)c1=O','CC(C)Cc1ccc(cc1)C(C)C(=O)O','CC(=O)Nc1ccc(O)cc1','O=C(O)c1ccccc1']
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)
 for d in ['input','output','workspace','present']:(p/d).mkdir()
 (p/'input/python_config.json').write_text(json.dumps({'proteinName':'fixture','isUseeToxPred':True,'saThreshold':10}))
 def fixture_run(command,**kwargs):
  if command[0]=='obabel':(p/'workspace/ligand.pdbqt').write_text('mocked docking input\n')
  else:kwargs['stdout'].write('HEADER\n1 -7.5 0 0\nEND\n');kwargs['stdout'].flush()
  return None
 with patch('add_node_type.subprocess.run',side_effect=fixture_run):
  scores=check_node_type(smiles.copy(),str(p)+'/')
(out/'generation.json').write_text(json.dumps({'sampling':generation,'docking_fixture_result':scores},indent=2,default=lambda value: value.tolist() if hasattr(value,'tolist') else value.item()));print('Done',len(generation),'sampling cases; fixture accepted:',len(scores[0]))
