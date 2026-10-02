# joblib security update validation

The update from joblib 1.1.0 to 1.2.0 fixes CVE-2022-21797
([GHSA-6hrg-qmvc-2xh8](https://github.com/advisories/GHSA-6hrg-qmvc-2xh8)):
`Parallel(pre_dispatch=...)` previously evaluated arbitrary Python expressions.
1.2.0 restricts evaluation to arithmetic expressions. Model loading with joblib
still requires trusted input; this update does not make pickle files safe.

## Reproduce the updated-version checks

Create a new environment; do not install into an existing research environment.
From the repository root, using Python 3.9:

```sh
python3.9 -m venv /tmp/mothra-joblib-validation
/tmp/mothra-joblib-validation/bin/python -m pip install \
  numpy==1.22.4 scipy==1.8.1 scikit-learn==1.1.1 \
  threadpoolctl==3.1.0 joblib==1.2.0 rdkit-pypi==2022.3.4 Pillow==9.2.0
/tmp/mothra-joblib-validation/bin/python -m unittest discover -s tests -v
/tmp/mothra-joblib-validation/bin/python -m pip check
```

`rdkit-pypi` is the historical PyPI distribution name used to obtain RDKit
2022.03.4 in this isolated validation environment. No RDKit pin was changed.

## Evidence

- Tested on macOS arm64, Python 3.9.25.
- NumPy 1.22.4, SciPy 1.8.1, scikit-learn 1.1.1, RDKit 2022.03.4,
  threadpoolctl 3.1.0 and Pillow 9.2.0 were held constant.
- The three compatibility checks passed with joblib 1.1.0 before the update,
  and again with 1.2.0. The fourth check specifically validates the security fix.
- `joblib_reference.json` stores results captured with 1.1.0 from the repository's
  actual `ligand_design/etoxpred_best_model.joblib`, plus SHA-256 hashes of the
  model and fingerprints. It is not generated from the updated version.
- For 32 SMILES, fingerprints follow `add_node_type.py`: explicit hydrogen,
  Morgan radius 2, 1024 floating-point bits. Both per-molecule and batched
  inference preserve toxicity probabilities, the `< 0.7` acceptance decisions
  (7 accepted, 25 rejected), and `1 - probability` scores exactly.
- Uncompressed and compressed model roundtrips and two-process inference
  preserve the baseline probabilities exactly.
- Arithmetic `pre_dispatch` expressions work; Python function-call expressions
  are rejected. OSV returned no advisories for joblib 1.2.0 at the time of the
  check on 2026-10-02. `pip check` passed in the isolated validation environment.

## Limits

The complete requirements file was not installed. These tests isolate the
changed dependency and its eToxPred/scikit-learn path. Full TensorFlow/CUDA RNN
training, molecular generation, external AutoDock Vina/Open Babel docking,
and Linux GPU container execution were not tested. Other dependency
vulnerabilities remain and should be addressed in separate updates.
