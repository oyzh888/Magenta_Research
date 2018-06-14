'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential,Model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Input,Flatten
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import random
import sys
import io

if(len(sys.argv)!=1):
    LSTM_Layer_Num = int(sys.argv[1])
else:
    LSTM_Layer_Num = 2

epochs = 100
def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-1
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)

    lr = 1e-3
    return lr

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

how_much_part = 1
text = text[:int(len(text)/how_much_part)]
print('truncated corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 25
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
# from IPython import embed; embed()
# import ipdb;
# ipdb.set_trace()


# build the model:  LSTM
print('Build model...')
LSTM_Layer_Num = 10
# 这部分返回一个张量
inputs = Input(shape=(maxlen, len(chars),))

# 层的实例是可调用的，它以张量为参数，并且返回一个张量
xxx = LSTM(128, return_sequences=True)(inputs)
for i in range(LSTM_Layer_Num - 1):
    if(i == LSTM_Layer_Num - 2):
        xxx = LSTM(128, return_sequences=False,dropout=0)(xxx)
    else:
        xxx = LSTM(128, return_sequences=True,dropout=0)(xxx)
# xxx = Flatten(xxx)
xxx = Dense(len(chars))(xxx)
predictions = Activation('softmax')(xxx)
# model = Sequential()
# shape=(32,)
# model.add(Input(shape=(maxlen, len(chars))))
# model.add(LSTM(512, input_shape=(maxlen, len(chars)),return_sequences=False))
# model.add(LSTM(128, input_shape=(maxlen, len(chars)),dropout=0.2, recurrent_dropout=0.2,
#                return_sequences=False))
# for i in range(LSTM_Layer_Num):
#     if(i==LSTM_Layer_Num-1):
#         model.add(LSTM(128, return_sequences=False))
#     else:
#         model.add(LSTM(128, return_sequences=True))
#     # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))
model = Model(inputs=inputs, outputs=predictions)
model.summary()
optimizer = RMSprop(lr=lr_schedule(0))

from keras import metrics

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    if(epoch%20 == 0):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
lr_scheduler = LearningRateScheduler(lr_schedule)

# 参照下面代码加一下TensorBoard
from keras.callbacks import TensorBoard
# tb_callbacks = TensorBoard(log_dir = './TB_logdir/LSTM/RNNStructure/512_LS')
tb_callbacks = TensorBoard(log_dir = './TB_logdir/LSTM/RNNStructure/%dLayer_LS128_BS512' % LSTM_Layer_Num)
model.fit(x, y,
          batch_size=512,
          epochs=epochs,
          callbacks=[print_callback,tb_callbacks,lr_scheduler])


