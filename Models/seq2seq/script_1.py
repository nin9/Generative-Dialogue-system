import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import keras
from gensim.models.word2vec import Word2Vec
from nltk import word_tokenize
import re

w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab_dim = 300
max_length = 200

NUMBER_OF_TRAINING_SAMPLES = 2000

clean_data_file = '../../Data/conv-clean.txt'

data = None
data_string = ''

with open(clean_data_file, mode='r') as f:
    data_string = f.read().strip().lower()
    data = data_string.split('\n')

NUMBER_OF_TRAINING_SAMPLES = len(data)

x1 = []
y1 = []
tokens = []

for line in data:
    if not re.match('.*\\t.*', line): continue
    s1 = line.strip().split('\t')
    x1.append(s1[0])
    y1.append(s1[1])
    tokens.extend(word_tokenize(line))

v = set(tokens)

vocab = list(v)

for token in vocab:
    if (not token in w2v):
        vocab.remove(token)

vocab.append('unk')

index_to_word = vocab
word_to_index = {word: index for index, word in enumerate(vocab)}

X = np.zeros((NUMBER_OF_TRAINING_SAMPLES, max_length, 300), dtype=np.float32)
Y = np.zeros((NUMBER_OF_TRAINING_SAMPLES, max_length, len(vocab)), dtype=np.bool)

non_existing = 0

for i, sen in enumerate(x1):
    for j, token in enumerate(word_tokenize(sen)):
        X[i, j, :] = w2v[token] if w2v.vocab.has_key(token) else w2v['unk']
        if not w2v.vocab.has_key(token):
            non_existing += 1

print('Number of non-existing tokens in input = ' + str(non_existing))

non_existing = 0

for i, sen in enumerate(y1):
    for j, token in enumerate(word_tokenize(sen)):
        if w2v.vocab.has_key(token):
            Y[i, j, word_to_index[token]] = 1
        else:
            Y[i, j, word_to_index['unk']] = 1
            non_existing += 1

print('Number of non-existing tokens in output = ' + str(non_existing))

# model = SimpleSeq2seq(input_dim=vocab_dim, hidden_dim=300, output_length=max_length, output_dim=len(vocab), depth=2)

hidden_dim = 128
model = Sequential()

# model.add(Embedding(len(vocab), vocab_dim, input_length=max_length))
# model.add(Dropout(0.20))

model.add(LSTM(hidden_dim, return_sequences=True, input_dim=300))
model.add(Dropout(0.20))

model.add(LSTM(hidden_dim, return_sequences=True))
model.add(Dropout(0.20))

# model.add(LSTM(hidden_dim, return_sequences=True))
# model.add(Dropout(0.20))
#
# model.add(LSTM(hidden_dim, return_sequences=True))
# model.add(Dropout(0.20))

model.add(TimeDistributed(Dense(len(vocab), activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights('model-weights.h5')
model.fit(X, Y, nb_epoch=10, batch_size=128)
model.save_weights('model-weights.h5', overwrite=True)

test = np.zeros((1, max_length, 300), dtype=np.float32)
test[0, :, :] = X[0, :, :]

preds = model.predict(test)[0, :, :]
out = ''
for w in preds:
    i = np.argmax(w)
    out += index_to_word[i] + ' '

print(out)
