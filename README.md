# Mothra

## Requirements
1. [Python](https://www.anaconda.com/download/)==3.9, some errors show up in newer version of python. NOTICE:`init.sh` uses pyenv, if you have not installed it, you should install `python3.9-venv`
2. [Keras](https://github.com/fchollet/keras) (version 2.0.5) If you installed the newest version of keras, some errors will show up. Please change it back to keras 2.0.5 by pip install keras==2.0.5. 
3. (*Optional but Highly recommended) [CUDA](https://developer.nvidia.com/cuda-downloads) (version 11.7) , [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (version 8 for CUDA 11.x)
3. tensoflow-gpu (version 1.15.2, ver>=2.0 occurred error.) 
4. [rdkit](https://anaconda.org/rdkit/rdkit)
5. [rDock](http://rdock.sourceforge.net/installation/)
6. [Autodock Vina](https://vina.scripps.edu/) Make sure to add Vina into system path.
7. [Open Babel](http://openbabel.org/wiki/Category:Installation) Make sure to add OpenBabel into system path.
8. [eToxPred](http://github.com/pulimeng/eToxPred) DL and untar https://github.com/pulimeng/eToxPred/raw/master/etoxpred_best_model.tar.gz into ligand_design/ for using toxcity prediction. Pretrained model is provided in `ligand_design` dir.

For installing Keras, rdkit, and other dependencies by `pip` on Virtual ENVironment, We provide `requirements.txt` and `init.sh` in `init` dir. After installing python, you may run `bash inits/init.sh`.

## How to Use
#### Install Docker
1. Get installer in https://docs.docker.com/engine/install/
1. Run `docker build -t hoge .` with CUDA GPU devices. `hoge` is a label for the docker containers.

#### Train the RNN model

1. Run `docker run --gpus all --rm -it -v .:/mnt:rw hoge python train_RNN/train_RNN.py` to train the RNN model. Pretrained model is provided in `model/model.h5`

#### Molecule generate

1. Run `docker run --gpus all --rm -it -v .:/mnt:rw hoge python ligand_design/mcts_ligand.py ./template_for_data/`

Although MOMCTS-MolGen has an extendable objective set, the default setting of objectives is docking score, QED score, logP, and a filter on SA score.

To modify your own objective set, change simulation functions in add_node_type.py, and change reward functions in mcts_ligand.py. (it may integrate into one function in future work)

If the size of the objective set is not 3, don't forget to change 'default_reward' in mcts_ligand.py.

Outputs of ligand_design process will store in data/present/, including:
```
output.txt             ## output of pareto front change
ligands.txt            ## ligands pass SA score filter.
scores.txt             ## raw scores of ligands
hverror_output.txt     ## output of hypervolume calculation errors
error_output.txt       ## output of vina and obabel errors
```


#### directory structure

```
.
├─data : for pretrain dataset
├─template_for_data : template directory for ligand generation
│  ├─input : set target protein(s) for docking on VINA and configure generation
│  ├─output : save generated ligands
│  ├─present : save valid generated ligands and their scores
│  └─workspace : a room for docking on each ligand
├─ligand_design : source code for ligand generation
├─model : save an RNN generative model.
└─train_RNN : train an RNN generative model.
```


## License
This package is distributed under the GPL License.
