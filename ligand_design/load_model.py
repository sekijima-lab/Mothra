import json
from pathlib import Path
import keras
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mothra_runtime import CompatibleGRU, set_model_mode

#from keras.models import load_model

def prepare_data(smiles,all_smile):
    all_smile_index=[]
    for i in range(len(all_smile)):
        smile_index=[]
        for j in range(len(all_smile[i])):
            smile_index.append(smiles.index(all_smile[i][j]))
        all_smile_index.append(smile_index)
    X_train=all_smile_index
    y_train=[]
    for i in range(len(X_train)):

        x1=X_train[i]
        x2=x1[1:len(x1)]
        x2.append(0)
        y_train.append(x2)

    return X_train,y_train


def loaded_model(rnnModelDir, old_compatible=True):
    directory = Path(rnnModelDir)
    if (directory / 'model.keras').exists():
        model=keras.models.load_model(directory / 'model.keras', compile=False, safe_mode=True)
        return set_model_mode(model,old_compatible)
    config = json.loads((directory / 'model.json').read_text())
    model = legacy_model_from_config(config,old_compatible)
    model.load_weights(directory / 'model.h5')
    return model


def legacy_model_from_config(document,old_compatible=True):
    """Reconstruct the historical sequential GRU graph using built-in layers only."""
    if document.get('class_name') != 'Functional':
        raise ValueError('Expected a legacy Functional model')
    graph = document['config']
    layers = graph['layers']
    if not layers or layers[0]['class_name'] != 'InputLayer':
        raise ValueError('Expected an input layer')
    cfg = layers[0]['config']
    if cfg.get('sparse') or cfg.get('ragged'):
        raise ValueError('Sparse/ragged legacy models are unsupported')
    inputs = keras.Input(shape=tuple(cfg['batch_input_shape'][1:]), dtype=cfg.get('dtype','float32'), name=cfg['name'])
    x, previous = inputs, layers[0]['name']
    allowed = {'Embedding':keras.layers.Embedding,'GRU':CompatibleGRU,
               'Dropout':keras.layers.Dropout,'TimeDistributed':keras.layers.TimeDistributed,'Dense':keras.layers.Dense}

    def build(entry):
        kind = entry['class_name']
        if kind not in allowed:
            raise ValueError('Unsupported legacy layer: '+kind)
        args = dict(entry['config'])
        args.pop('batch_input_shape',None)
        args.pop('input_length',None)
        if args.pop('time_major',False):
            raise ValueError('time_major=True is unsupported')
        if args.pop('implementation',2) != 2:
            raise ValueError('Only the historical implementation=2 is supported')
        for key,value in list(args.items()):
            if key.endswith('_initializer') and isinstance(value,dict):
                if value['class_name'] not in {'RandomUniform','GlorotUniform','Orthogonal','Zeros'}:
                    raise ValueError('Unsupported initializer')
                args[key]={'class_name':value['class_name'],'config':value['config']}
            if key.endswith('_regularizer') or key.endswith('_constraint'):
                if value is not None:
                    raise ValueError('Custom regularizers/constraints are unsupported')
        if args.get('activation') not in (None,'tanh','sigmoid','softmax','linear') or args.get('recurrent_activation') not in (None,'sigmoid'):
            raise ValueError('Unsupported activation')
        if kind == 'GRU':
            if args.get('activation') != 'tanh' or args.get('recurrent_activation') != 'sigmoid':
                raise ValueError('Compatibility mode supports only tanh/sigmoid GRUs')
            args['old_compatible']=old_compatible
        if kind == 'TimeDistributed':
            args['layer'] = build(args['layer'])
        return allowed[kind](**args)

    for entry in layers[1:]:
        if entry['inbound_nodes'] != [[[previous,0,0,{}]]]:
            raise ValueError('Only a single-chain legacy graph is supported')
        x = build(entry)(x)
        previous = entry['name']
    if graph['input_layers'] != [[layers[0]['name'],0,0]] or graph['output_layers'] != [[previous,0,0]]:
        raise ValueError('Invalid graph endpoints')
    return keras.Model(inputs=inputs, outputs=x, name=graph.get('name','model'))
