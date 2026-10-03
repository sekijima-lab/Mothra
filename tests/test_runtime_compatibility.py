"""Historical probabilities and explicit runtime mode must survive archiving."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import tensorflow as tf
import keras
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'ligand_design'))
from mothra_runtime import CompatibleGRU, add_runtime_arguments, legacy_sigmoid, legacy_tanh
from load_model import loaded_model,legacy_model_from_config


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline=np.load(ROOT/'tests/runtime_reference.npz',allow_pickle=False)

    def test_default_mode_matches_legacy_and_standard_is_available(self):
        compatible=loaded_model(ROOT/'model3')
        actual=compatible(self.baseline['inputs'],training=False).numpy()
        np.testing.assert_allclose(actual,self.baseline['probabilities'],atol=1e-5,rtol=0)
        np.testing.assert_array_equal(actual.argmax(-1),self.baseline['probabilities'].argmax(-1))
        native=loaded_model(ROOT/'model3',old_compatible=False)
        self.assertTrue(all(l.old_compatible for l in compatible.layers if isinstance(l,CompatibleGRU)))
        self.assertTrue(all(not l.old_compatible for l in native.layers if isinstance(l,CompatibleGRU)))
        native_result=native(self.baseline['inputs'],training=False).numpy()
        self.assertGreater(float(np.max(abs(actual-native_result))),0)
        for a,b in zip(compatible.weights,native.weights):np.testing.assert_array_equal(a.numpy(),b.numpy())

    def test_archive_preserves_modes_and_explicit_override(self):
        for mode in [True,False]:
            model=loaded_model(ROOT/'model3',old_compatible=mode)
            expected=model(self.baseline['inputs'][:2],training=False).numpy()
            with tempfile.TemporaryDirectory() as directory:
                model.save(Path(directory)/'model.keras')
                archive=keras.models.load_model(Path(directory)/'model.keras',safe_mode=True,compile=False)
                self.assertTrue(all(l.old_compatible == mode for l in archive.layers if isinstance(l,CompatibleGRU)))
                np.testing.assert_array_equal(archive(self.baseline['inputs'][:2],training=False).numpy(),expected)
                loaded=loaded_model(directory,old_compatible=mode)
                np.testing.assert_array_equal(loaded(self.baseline['inputs'][:2],training=False).numpy(),expected)
                override=loaded_model(directory,old_compatible=not mode)
                self.assertTrue(all(l.old_compatible != mode for l in override.layers if isinstance(l,CompatibleGRU)))

    def test_activation_extremes_and_gradients(self):
        values=tf.constant([-float('inf'),-1000.,-104.,-100.,-90.,-20.,0.,20.,88.,89.,1000.,float('inf')])
        for fn in [legacy_sigmoid,legacy_tanh]:
            with tf.GradientTape() as tape:
                tape.watch(values);result=fn(values)
            gradient=tape.gradient(result,values)
            self.assertTrue(np.isfinite(result.numpy()).all())
            self.assertTrue(np.isfinite(gradient.numpy()).all())
            np.testing.assert_array_equal(tf.function(fn)(values).numpy(),result.numpy())
        sigmoid=legacy_sigmoid(values).numpy()
        self.assertEqual(sigmoid[0],0);self.assertEqual(sigmoid[-1],1);self.assertEqual(sigmoid[6],.5)

    def test_unapproved_legacy_layer_rejected(self):
        config=json.loads((ROOT/'model3/model.json').read_text())
        config['config']['layers'][1]['class_name']='Lambda'
        with self.assertRaisesRegex(ValueError,'Unsupported legacy layer'):legacy_model_from_config(config)
        config=json.loads((ROOT/'model3/model.json').read_text())
        next(l for l in config['config']['layers'] if l['class_name']=='GRU')['config']['activation']='linear'
        with self.assertRaisesRegex(ValueError,'tanh/sigmoid'):legacy_model_from_config(config)

    def test_cli_modes(self):
        parser=argparse.ArgumentParser();add_runtime_arguments(parser)
        self.assertTrue(parser.parse_args([]).old_compatible)
        self.assertTrue(parser.parse_args(['--old-compatible']).old_compatible)
        self.assertFalse(parser.parse_args(['--no-old-compatible']).old_compatible)

if __name__=='__main__':unittest.main()
