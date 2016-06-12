from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras

import matplotlib.pyplot as plt

f = open('../Data/data.txt','r')
s = f.read()
total_text = s

chars = set(total_text)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 20
step = 1
sentences = []
next_chars = []
for i in range(0, len(total_text) - maxlen, step):
    sentences.append(total_text[i: i + maxlen])
    next_chars.append(total_text[i + maxlen])
print('nb sequences:', len(sentences))

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_dim=len(chars)))
model.add(Dropout(0.20))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.20))

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.20))

model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.20))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

checkpoint = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=True, save_best_only=False, mode='auto')
hist = keras.callbacks.History()

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights('script_1_weights.h5')

model.fit(X, y, nb_epoch=1, callbacks=[checkpoint, hist], batch_size=1024)

model.save_weights('script_1_weights.h5', overwrite=True)

seed_text = 'hello, there'
generated = ''
for iteration in range(50):
    x = np.zeros((1, len(seed_text), len(chars)))
    for t, char in enumerate(seed_text):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = np.argmax(preds)
    next_char = indices_char[next_index]

    generated += next_char
    seed_text = seed_text[1:] + next_char
print(generated)

plt.plot(range(len(hist.history['loss'])), hist.history['loss'])
plt.show()
