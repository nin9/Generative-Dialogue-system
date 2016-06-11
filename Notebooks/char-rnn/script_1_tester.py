from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras

f = open('../Data/data.txt','r')
s = f.read()
total_text = s

chars = set(total_text)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_dim=len(chars)))
model.add(Dropout(0.20))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.20))

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.20))

model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.20))

model.add(Dense( len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights('weights.02-1.33.hdf5')

while True:
    print('Enter seed text:')
    seed_text = raw_input()
    generated = ''
    next_char = ''
    for iteration in range(50):
        # if next_char == '\n': break
        x = np.zeros((1, len(seed_text), len(chars)))
        for t, char in enumerate(seed_text):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]

        generated += next_char
        seed_text = seed_text[1:] + next_char
    print(generated)
