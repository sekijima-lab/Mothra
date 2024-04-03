import csv
import datetime
import json
import numpy as np
import os
from keras.layers import Embedding,Dense, Activation,TimeDistributed, GRU, Dropout, Input
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, EarlyStopping
from make_smile import zinc_data_with_bracket_original,zinc_processed_with_bracket

from keras.utils import pad_sequences

from tensorflow import distribute, data
import tensorflow as tf


def load_data():

    sen_space=[]
    f = open(os.path.dirname(__file__)+'/../data/smile_trainning.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        sen_space.append(row)
    f.close()

    element_table=["Cu","Ti","Zr","Ga","Ge","As","Se","Br","Si","Zn","Cl","Be","Ca","Na","Sr","Ir","Li","Rb","Cs","Fr","Be","Mg",
            "Ca","Sr","Ba","Ra","Sc","La","Ac","Ti","Zr","Nb","Ta","Db","Cr","Mo","Sg","Mn","Tc","Re","Bh","Fe","Ru","Os","Hs","Co","Rh",
            "Ir","Mt","Ni","Pd","Pt","Ds","Cu","Ag","Au","Rg","Zn","Cd","Hg","Cn","Al","Ga","In","Tl","Nh","Si","Ge","Sn","Pb","Fl",
            "As","Sb","Bi","Mc","Se","Te","Po","Lv","Cl","Br","At","Ts","He","Ne","Ar","Kr","Xe","Rn","Og"]
    word1=sen_space[0]
    word_space=list(word1[0])
    end="\n"
    word_space.append(end)
    all_smile=[]
    for i in range(len(sen_space)):
        word1=sen_space[i]
        word_space=list(word1[0])
        word=[]
        j=0
        while j<len(word_space):
            word_space1=[]
            word_space1.append(word_space[j])
            if j+1<len(word_space):
                word_space1.append(word_space[j+1])
                word_space2=''.join(word_space1)
            else:
                word_space1.insert(0,word_space[j-1])
                word_space2=''.join(word_space1)
            if word_space2 not in element_table:
                word.append(word_space[j])
                j=j+1
            else:
                word.append(word_space2)
                j=j+2

        word.append(end)
        all_smile.append(list(word))
    val=[]
    for i in range(len(all_smile)):
        for j in range(len(all_smile[i])):
            if all_smile[i][j] not in val:
                val.append(all_smile[i][j])
    val.remove("\n")
    val.insert(0,"\n")

    return val, all_smile


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

"""
#UNUSED CODE.
def generate_smile(model,val):
    end="\n"
    start_smile_index= [val.index("C")]
    new_smile=[]

    while not start_smile_index[-1] == val.index(end):
        predictions=model.predict(start_smile_index)
        ##next atom probability
        smf=[]
        for i in range (len(X)):
            sm=[]
            for j in range(len(X[i])):
                #if np.argmax(predictions[i][j])=!0
                sm.append(np.argmax(predictions[i][j]))
            smf.append(sm)

        #print sm
        #print smf
        #print len(sm)

        new_smile.append(sampled_word)
    #sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    #return new_sentence
"""


def save_model(model):
    """
        Save model by JSON and keras scheme.
    """
    # serialize model to JSON
    model_json = model.to_json(indent=4, separators=(',', ': '))
    with open("model2.json", "w") as json_file:
         json_file.write(model_json)
    # serialize weights to HDF5
    model.save("model.h5",save_format='h5')
    model.save("model",save_format='tf')
    print("Saved model to disk")


def _createModel(vocab_size: int, embed_size: int, N: int):
    """
        Create NN Model.
    """
    input = Input(shape=(N,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=N, mask_zero=False)(input)
    x = GRU(units=256,activation='tanh',return_sequences=True)(x)
    x = Dropout(.2)(x)
    x = GRU(units=256,activation='tanh',return_sequences=True)(x)
    x = Dropout(.2)(x)
    x = TimeDistributed(Dense(embed_size, activation='softmax'))(x)
    model = Model(inputs=input, outputs=x)

    return model

class EarlyStoppingByTimer(Callback):
    """
        Stop training when the loss is at its min, i.e. the loss stops decreasing.

        Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """
    def __init__(self, patience=0, startTime=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=+9),'JST')), timeLimit=datetime.timedelta(hours=23)):
        super(EarlyStoppingByTimer, self).__init__()
        self._time = startTime
        self._timeLimit = timeLimit - datetime.timedelta(minutes=1)
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self._recentTrainBegin = datetime.timedelta()
        self._JST = datetime.timezone(datetime.timedelta(hours=+9),'JST')

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        self._recentTrainBegin = datetime.datetime.now(self._JST)
        return super().on_epoch_begin(epoch, logs)
        
    def on_epoch_end(self, epoch, logs=None):
        _now = datetime.datetime.now(self._JST)
        _recentTimeDelta = _now - self._recentTrainBegin

        if(_now + _recentTimeDelta) >= (self._timeLimit + self._time):
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Saving Models in This JOB")
                #self.model.set_weights(self.best_weights)
                self.on_train_end()
        #current = logs.get("loss")
        #if np.less(current, self.best):
        #    self.best = current
        #    self.wait = 0
        #    # Record the best weights if current results is better (less).
        #    self.best_weights = self.model.get_weights()


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

if __name__ == "__main__":
    startTime = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=+9),'JST'))
    # set up for multi-gpu env.
    dataOpt = data.Options()
    dataOpt.experimental_distribute.auto_shard_policy = data.experimental.AutoShardPolicy.DATA
    strategy = distribute.MirroredStrategy()

    if os.path.exists('./train_RNN/config.json') :
        # load configuares.
        config = json.load(open('./train_RNN/config.json','r'))
        batchSize = config['batchSize']
        learningRate = config['learningRate']
        trainingSplit = 1.0 - config['validationSplit']
        earlystoppingByTimer = EarlyStoppingByTimer(
            startTime=startTime,
            timeLimit=datetime.timedelta(
                hours=config['limitTimerHours'],
                minutes=config['limitTimerMinutes'],
                seconds=config['limitTimerSeconds'])
            )
        init = config['last_epoch']
        isLoadWeight = config['isLoadWeight']
        whereisWeightFile = config['whereisWeightFile']
        needTensorboard = config['needTensorboard']
    else:
        # set default
        batchSize = 64
        learningRate = .01
        trainingSplit = 1.0 - .1
        earlystoppingByTimer = EarlyStoppingByTimer(
            startTime=startTime,
            timeLimit=datetime.timedelta(
                hours=2)
            )
        init = 0
        isLoadWeight = false
        needTensorboard = false
    GLOBAL_BATCH_SIZE = batchSize * strategy.num_replicas_in_sync

    # prepare data from /data
    smile=zinc_data_with_bracket_original()
    valcabulary,all_smile=zinc_processed_with_bracket(smile)
    X_train,y_train=prepare_data(valcabulary,all_smile)
  
    maxlen=81

    X = pad_sequences(X_train, maxlen=81, dtype='int32',
        padding='post', truncating='pre', value=0.)
    y = pad_sequences(y_train, maxlen=81, dtype='int32',
        padding='post', truncating='pre', value=0.)
    
    y_train_one_hot = np.array([to_categorical(sent_label, num_classes=len(valcabulary)) for sent_label in y])
    vocab_size=len(valcabulary)
    embed_size=len(valcabulary)
    N=X.shape[1]

    X_nd_train, X_nd_valid = X[:int(len(X)*trainingSplit)], X[int(len(X)*trainingSplit):]
    y_nd_train_one_hot, y_nd_valid_one_hot = y_train_one_hot[:int(len(y_train_one_hot)*trainingSplit)], y_train_one_hot[int(len(y_train_one_hot)*trainingSplit):]
    
    trainDataset = data.Dataset.zip((data.Dataset.from_tensor_slices(X_nd_train), data.Dataset.from_tensor_slices(tf.cast(y_nd_train_one_hot, dtype=tf.float32)))).shuffle(buffer_size=N).batch(GLOBAL_BATCH_SIZE).with_options(dataOpt)
    validDataset = data.Dataset.zip((data.Dataset.from_tensor_slices(X_nd_valid), data.Dataset.from_tensor_slices(tf.cast(y_nd_valid_one_hot, dtype=tf.float32))))                       .batch(GLOBAL_BATCH_SIZE).with_options(dataOpt)

    # For MirroredStrategy, to move data to device mem.
    # trainDistDataset = strategy.experimental_distribute_dataset(trainDataset)
    # validDistDataset = strategy.experimental_distribute_dataset(validDataset)

    # make up callbacks
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)
    callbacks = [earlystoppingByTimer]#, earlyStopping]
    if needTensorboard:
        tensorboard_callback = TensorBoard(log_dir="../tensorboard_logs", profile_batch=5)
        callbacks.append(tensorboard_callback)
    # prepare custom loss function
    with strategy.scope():
        def compute_loss(labels, predictions):
            loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        if isLoadWeight:
            model = tf.keras.models.load_model(os.path.join(os.path.curdir,whereisWeightFile))
        else:
            model = _createModel(vocab_size=vocab_size,embed_size=embed_size,N=N)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), metrics=['accuracy'])
        model.fit(x=trainDataset,epochs=100, validation_data=validDataset, callbacks=callbacks,initial_epoch=init,verbose=2)
        #model.fit(X,y_train_one_hot,epochs=100, validation_split=.1, callbacks=callbacks,initial_epoch=init,verbose=2)
        
    save_model(model)

    # Save Stopped Epoch ton config.json
    if os.path.exists('./train_RNN/config.json'):
        config = json.load(open('./train_RNN/config.json','r'))
        config["last_epoch"] = earlystoppingByTimer.stopped_epoch
        json.dump(config,open('./train_RNN/config.json','w'), indent=4, separators=(',', ': '))