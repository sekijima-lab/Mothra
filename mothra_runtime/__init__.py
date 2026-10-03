"""Versioned Mothra model runtime and explicit historical compatibility mode."""
import argparse
import json
import platform
import importlib.metadata
import keras
from .activations import legacy_tanh, legacy_sigmoid
from .optimizers import DeduplicatingAdam

@keras.saving.register_keras_serializable(package="Mothra")
class CompatibleGRU(keras.layers.GRU):
    def __init__(self, units, old_compatible=True, **kwargs):
        self.old_compatible=bool(old_compatible)
        if self.old_compatible:
            kwargs['activation']=legacy_tanh
            kwargs['recurrent_activation']=legacy_sigmoid
            kwargs['use_cudnn']=False
        else:
            kwargs['activation']='tanh'
            kwargs['recurrent_activation']='sigmoid'
            kwargs['use_cudnn']='auto'
        super().__init__(units,**kwargs)
        if self.old_compatible and self.compute_dtype != 'float32':
            raise ValueError('Old compatibility requires float32 GRUs')

    def get_config(self):
        config=super().get_config()
        config['old_compatible']=self.old_compatible
        return config


def set_model_mode(model, old_compatible=True):
    """Clone GRU layers when an explicit execution mode differs from the archive."""
    grus=[layer for layer in model.layers if isinstance(layer,keras.layers.GRU)]
    if not grus:
        raise ValueError('Expected an RNN containing GRU layers')
    for layer in grus:
        if layer.cell.activation not in (legacy_tanh,keras.activations.tanh) or layer.cell.recurrent_activation not in (legacy_sigmoid,keras.activations.sigmoid):
            raise ValueError('Compatibility mode supports only tanh/sigmoid GRUs')
    if all(isinstance(layer,CompatibleGRU) and layer.old_compatible == old_compatible for layer in grus):
        return model
    def clone(layer):
        if isinstance(layer,keras.layers.GRU):
            config=layer.get_config()
            config.pop('old_compatible',None)
            return CompatibleGRU(old_compatible=old_compatible,**config)
        return layer.__class__.from_config(layer.get_config())
    converted=keras.models.clone_model(model,clone_function=clone)
    converted.set_weights(model.get_weights())
    return converted


def add_runtime_arguments(parser):
    parser.add_argument('--old-compatible',action=argparse.BooleanOptionalAction,default=True,
                        help='Historical CPU activations and Adam sparse-gradient semantics (default: enabled)')


def runtime_info(old_compatible):
    names=['tensorflow','keras','numpy','scipy','pandas','h5py','rdkit','joblib','moocore','tensorboard']
    return {'old_compatible':bool(old_compatible),'python':platform.python_version(),
            'libraries':{name:importlib.metadata.version(name) for name in names}}


def log_runtime(old_compatible):
    print('Mothra runtime: '+json.dumps(runtime_info(old_compatible),sort_keys=True),flush=True)
