"""Compare independently recorded baseline and candidate arrays; enforce frozen limits."""
import argparse,json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('baseline',type=Path);parser.add_argument('candidate',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
criteria=json.loads(Path(__file__).with_name('criteria.json').read_text());results={}
def check(name,actual,expected,limit):
 difference=float(np.max(abs(actual-expected)));results[name]={'max_abs':difference,'limit':limit,'pass':difference<=limit}
a=np.load(args.baseline/'prediction.npz');b=np.load(args.candidate/'prediction.npz')
check('probability',b['probabilities'],a['probabilities'],criteria['rnn_probability_max_abs'])
mae=float(np.mean(abs(b['probabilities']-a['probabilities'])));results['probability_mae']={'mean_abs':mae,'limit':criteria['rnn_probability_mae'],'pass':mae<=criteria['rnn_probability_mae']}
for name,key in [('inputs','inputs'),('targets','targets')]:results[name]={'pass':np.array_equal(a[key],b[key])}
results['argmax']={'pass':np.array_equal(a['probabilities'].argmax(-1),b['probabilities'].argmax(-1))}
for file,prefix,section,limit in [('initial-weights.npz','weight','initial_weights',0),('training.npz','gradient','gradients',criteria['gradient_max_abs']),('training.npz','weight','updated_weights',criteria['adam_weight_max_abs'])]:
 a=np.load(args.baseline/file);b=np.load(args.candidate/file);values={k:float(np.max(abs(a[k]-b[k]))) for k in a.files if k.startswith(prefix)}
 results[section]={'per_tensor':values,'max_abs':max(values.values()),'limit':limit,'pass':max(values.values())<=limit}
a=np.load(args.baseline/'toxicity.npz');b=np.load(args.candidate/'toxicity.npz')
results['fingerprints']={'pass':np.array_equal(a['fingerprints'],b['fingerprints'])}
check('toxicity',b['probabilities'],a['probabilities'],criteria['toxicity_on_identical_fingerprints_max_abs'])
results['toxicity_acceptance']={'pass':np.array_equal(a['probabilities'][:,1]<.7,b['probabilities'][:,1]<.7)}
check('qed',b['rewards'][:,0],a['rewards'][:,0],criteria['qed_max_abs']);check('sa',b['rewards'][:,1],a['rewards'][:,1],criteria['sa_score_max_abs'])
a=json.loads((args.baseline/'summary.json').read_text());b=json.loads((args.candidate/'summary.json').read_text())
check('loss',np.array(b['loss']),np.array(a['loss']),criteria['loss_abs']);results['vocabulary']={'pass':a['vocabulary']==b['vocabulary']}
results['source_hashes']={'pass':a['source_hashes']==b['source_hashes']}
results['pass']=all(v['pass'] for v in results.values())
args.output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
raise SystemExit(0 if results['pass'] else 1)
