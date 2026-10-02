"""Regression checks for the joblib security update (Python 3.9).

Run: python -m unittest discover -s tests -v
The JSON reference was captured with joblib 1.1.0 and all other dependencies
unchanged. This exercises eToxPred inference, not the full GPU/docking pipeline.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ligand_design" / "etoxpred_best_model.joblib"
REFERENCE = Path(__file__).with_name("joblib_reference.json")
SMILES = [
    "C", "CC", "CCC", "CCCC", "CCO", "CCCO", "CC(C)O", "CCN",
    "CC(=O)O", "CC(=O)N", "COC", "CCOC", "c1ccccc1", "Oc1ccccc1",
    "Nc1ccccc1", "Cc1ccccc1", "c1ccncc1", "c1ccoc1", "C1CCCCC1",
    "C1CCNCC1", "CCCl", "CCBr", "CC(F)F", "CS", "CCS", "CC#N",
    "O=C=O", "CC(=O)Oc1ccccc1C(=O)O", "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Nc1ccc(O)cc1", "O=C(O)c1ccccc1",
]


def fingerprints():
    # Matches add_node_type.py: explicit H, Morgan radius 2, 1024 float bits.
    rows = []
    for smiles in SMILES:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        rows.append(np.array(list(fp.ToBitString()), dtype=float))
    return np.array(rows)


def summarize(model, x):
    p = model.predict_proba(x)[:, 1]
    return {"probabilities": p.tolist(), "accepted": (p < 0.7).tolist(),
            "toxicity_scores": (1 - p).tolist()}


class JoblibCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text())
        cls.x = fingerprints()
        cls.model = joblib.load(MODEL)

    def test_bundled_model_matches_baseline(self):
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         self.reference["model_sha256"])
        self.assertEqual(hashlib.sha256(self.x.tobytes()).hexdigest(),
                         self.reference["fingerprints_sha256"])
        current = summarize(self.model, self.x)
        for field in ["probabilities", "toxicity_scores"]:
            np.testing.assert_array_equal(current[field], self.reference[field])
        self.assertEqual(current["accepted"], self.reference["accepted"])
        # The application calls predict_proba separately for each molecule.
        individual = [self.model.predict_proba(row.reshape(1, 1024))[0, 1]
                      for row in self.x]
        np.testing.assert_array_equal(individual, self.reference["probabilities"])

    def test_model_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for compress in [0, 3]:
                path = Path(directory) / (str(compress) + ".joblib")
                joblib.dump(self.model, path, compress=compress)
                restored = joblib.load(path)
                np.testing.assert_array_equal(restored.predict_proba(self.x)[:, 1],
                                              self.reference["probabilities"])

    def test_process_parallel_inference(self):
        results = joblib.Parallel(n_jobs=2)(
            joblib.delayed(self.model.predict_proba)(row.reshape(1, 1024))
            for row in self.x)
        np.testing.assert_array_equal([result[0, 1] for result in results],
                                      self.reference["probabilities"])

    def test_pre_dispatch_accepts_arithmetic_but_rejects_calls(self):
        self.assertEqual(joblib.Parallel(n_jobs=2, pre_dispatch="2 * n_jobs")(
            joblib.delayed(abs)(value) for value in [-3, -2, -1]), [3, 2, 1])
        # Harmless Python call syntax is enough to verify the security fix.
        with self.assertRaises(ValueError):
            joblib.Parallel(n_jobs=2, pre_dispatch="len([1])")(
                joblib.delayed(abs)(value) for value in [-1])


if __name__ == "__main__":
    unittest.main()
