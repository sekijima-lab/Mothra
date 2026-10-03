# Mothra

Molecular generation with an RNN, multi-objective tree search, QED/SA scoring,
AutoDock Vina docking and eToxPred toxicity filtering.

## Install

Use **Python 3.12.15** and a new environment. Do not install into an existing
research environment.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The runtime uses TensorFlow 2.21.0 / Keras 3.15.1 and RDKit 2026.03.6. The
bundled RNN was actually saved with Keras 2.9.0; the older README's Keras 2.0.5
and TensorFlow 1.15.2 instructions did not describe that saved model.
Install `vina` and `obabel` separately and make them available on PATH for real
docking. Preserve the external tool versions and receptor preparation when
reproducing a previous docking experiment.

## Historical compatibility mode

**Old compatibility is enabled by default**, for both inference and training.
It reproduces the historical CPU float32 sigmoid/tanh approximations and sums
repeated embedding-gradient indices before Adam's second-moment update. It
preserves the bundled GRU weights, architecture, vocabulary and dropout rates.
It uses TensorFlow operations and requires no native extension.

`--old-compatible` explicitly selects the default; `--no-old-compatible` uses
standard current Keras GRUs and Adam. The latter may use accelerated kernels
where available and gives slightly different results. Both commands log the
selected mode and core library versions. Saved `.keras` models contain the mode
in their GRU configuration; `runtime.json` records mode and runtime versions.
An explicit execution flag takes precedence over a saved mode.

Validation on macOS arm64 CPU: 256 molecules have maximum probability
difference **5.96e-8** from the historical runtime, with identical argmax tokens;
nine fixed-seed generation cases match exactly. These are finite comparisons,
not guarantees for all seeds, full stochastic training or GPU/CUDA execution.
See [validation and security evidence](tests/RUNTIME_VALIDATION.md).

## Train the RNN

Run commands from the repository root:

```sh
.venv/bin/python train_RNN/train_RNN.py --epochs 100
# Standard current Keras runtime, explicitly selected:
.venv/bin/python train_RNN/train_RNN.py --epochs 100 --no-old-compatible
```

The default output is `model3/model.keras`, `model3/model.weights.h5`,
`model3/model.json` and `model3/runtime.json`. The existing historical JSON/HDF5
pair remains available; saving into `model3` replaces its JSON description,
so use `--output another-directory` to retain the original pair unchanged.
Use only trusted model archives. Loading uses Keras `safe_mode=True`; the
historical JSON loader permits only the supported single-chain built-in graph.

For a small isolated check, copy `train_RNN/config.json` to a temporary path
and use `--config`, `--limit-data 64`, `--epochs 1`, `--output` and
`--tensorboard-dir`. The complete dataset still determines the vocabulary.
To resume, set `isLoadWeight=true`, `whereisWeightFile` to a trusted `.keras`
archive and `last_epoch` to the completed epoch count in the scratch config.
`--epochs` is the final epoch number. As in the historical program, recompiling
starts fresh optimizer state; this is a weight-based continuation.

## Generate molecules

```sh
.venv/bin/python ligand_design/mcts_ligand.py ./template_for_data/
# Optional standard runtime:
.venv/bin/python ligand_design/mcts_ligand.py ./template_for_data/ --no-old-compatible
```

Set `whereisRNNmodelDir` in the input configuration to a directory containing
`model.keras`, or the bundled legacy `model.json`/`model.h5` pair. The bundled
toxicity model is a numeric-only `ligand_design/etoxpred_model.npz`; no sklearn
pickle needs to be downloaded or loaded at runtime. The legacy model's source
hash and probability fixtures are retained in tests. An offline converter in
`tools/` is restricted to that trusted historical pickle and must be run in its
historical sklearn environment, not the new runtime.

The default three objectives and SA/toxicity filters remain unchanged. For
custom objectives, update simulation/reward functions in `add_node_type.py`
and `mcts_ligand.py`, including `default_reward` if dimensionality changes.
Outputs are written into the selected data directory's `present/` directory:
`output.txt`, `ligands.txt`, `scores.txt`, `hverror_output.txt`, and
`error_output.txt`.

## CPU container

```sh
docker build -t mothra .
docker run --rm -it -v "$PWD:/mnt" mothra python ligand_design/mcts_ligand.py ./template_for_data/
```

The Dockerfile now uses a digest-pinned Python 3.12.15 CPU image instead of the
historical CUDA 11.2/Python 3.9 image. Debian supplies Open Babel and Vina, whose
versions can differ from the historical container. **The image build and real
docking were not validated**, because the Docker daemon was unavailable. The
separate `viewer_up/` notebook container and GPU deployment are outside this
runtime migration's validation scope.

## Tests

```sh
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The HTTP tests use only localhost TLS, a test certificate, dummy OAuth values
and no real credentials. `openssl` must be available.

## License

Mothra retains its repository license. Adapted Eigen activation code in
`mothra_runtime/activations.py` is MPL-2.0; see its
[attribution notice](mothra_runtime/LICENSE-EIGEN-NOTICE.md).
