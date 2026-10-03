"""Repeated sparse embedding indices must be summed before Adam squares them."""
import sys
from pathlib import Path
import unittest
import numpy as np
import tensorflow as tf
import keras
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train_RNN'))
from optimizers import DeduplicatingAdam


class AdamCompatibilityTests(unittest.TestCase):
    def test_sparse_duplicates_equal_explicit_dense_gradient(self):
        sparse_variable = tf.Variable(np.zeros((4, 2), dtype=np.float32))
        dense_variable = tf.Variable(np.zeros((4, 2), dtype=np.float32))
        sparse_optimizer = DeduplicatingAdam(learning_rate=0.0001)
        dense_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        for indices, values in [([0,0,1], [[1.,2.],[-.5,-1.],[2.,3.]]),
                                ([1,2,2], [[-.5,.25],[3.,1.],[-2.,-.5]]),
                                ([3,3], [[1.,2.],[-1.,-2.]])]:
            gradient = tf.IndexedSlices(tf.constant(values),tf.constant(indices),tf.constant([4,2]))
            sparse_optimizer.apply_gradients([(gradient,sparse_variable)])
            dense_optimizer.apply_gradients([(tf.convert_to_tensor(gradient),dense_variable)])
            np.testing.assert_array_equal(sparse_variable.numpy(),dense_variable.numpy())
            for actual,expected in zip(sparse_optimizer.variables,dense_optimizer.variables):
                np.testing.assert_array_equal(actual.numpy(),expected.numpy())

    def test_optimizer_serialization(self):
        original=DeduplicatingAdam(learning_rate=.0001)
        restored=keras.optimizers.deserialize(keras.optimizers.serialize(original))
        self.assertIsInstance(restored,DeduplicatingAdam)
        self.assertEqual(restored.get_config(),original.get_config())

if __name__ == '__main__':
    unittest.main()
