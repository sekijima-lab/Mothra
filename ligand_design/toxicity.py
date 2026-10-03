"""Tensor-free, pickle-free inference for the bundled binary ExtraTrees model."""
from pathlib import Path
import numpy as np


class ToxicityPredictor:
    def __init__(self, path=None):
        path = Path(path) if path is not None else Path(__file__).with_name('etoxpred_model.npz')
        with np.load(path, allow_pickle=False) as data:
            if int(data['format_version']) != 1 or int(data['n_features']) != 1024:
                raise ValueError('Unsupported toxicity model format')
            self.classes_ = data['classes'].copy()
            self.offsets = data['tree_offsets'].copy()
            self.left = data['children_left'].copy()
            self.right = data['children_right'].copy()
            self.feature = data['feature'].copy()
            self.threshold = data['threshold'].copy()
            self.probabilities = data['probabilities'].copy()
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError('Expected binary class ordering [0, 1]')
        size = len(self.left)
        if self.offsets.ndim != 1 or len(self.offsets) < 2 or self.offsets[0] != 0 or self.offsets[-1] != size or np.any(np.diff(self.offsets) <= 0):
            raise ValueError('Invalid tree offsets')
        if any(a.shape != (size,) for a in [self.right, self.feature, self.threshold]) or self.probabilities.shape != (size, 2):
            raise ValueError('Invalid tree dimensions')
        if not np.isfinite(self.probabilities).all() or not np.isfinite(self.threshold).all() or np.any(self.probabilities < 0) or not np.allclose(self.probabilities.sum(1), 1):
            raise ValueError('Invalid tree values')
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            left, right = self.left[start:end], self.right[start:end]
            leaf = left == -1
            if not np.array_equal(leaf, right == -1):
                raise ValueError('Invalid leaf')
            internal = np.flatnonzero(~leaf)
            if np.any(left[internal] <= internal) or np.any(right[internal] <= internal) or np.any(left[internal] >= end-start) or np.any(right[internal] >= end-start):
                raise ValueError('Invalid tree edge')
            features = self.feature[start:end][internal]
            if np.any(features < 0) or np.any(features >= 1024):
                raise ValueError('Invalid feature index')

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 1024 or not np.isfinite(X).all():
            raise ValueError('Expected finite (n, 1024) fingerprints')
        result = np.zeros((len(X), 2), dtype=np.float64)
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            nodes = np.zeros(len(X), dtype=np.int64)
            while True:
                active = np.flatnonzero(self.left[start + nodes] != -1)
                if not len(active):
                    break
                indices = start + nodes[active]
                go_left = X[active, self.feature[indices]] <= self.threshold[indices]
                nodes[active] = np.where(go_left, self.left[indices], self.right[indices])
            result += self.probabilities[start + nodes]
        return result / (len(self.offsets) - 1)
