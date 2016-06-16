import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras
from gensim.models.word2vec import Word2Vec
from nltk import word_tokenize
import seq2seq
from seq2seq.models import SimpleSeq2seq
import re

w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab_dim = 300
max_length = 100

NUMBER_OF_TRAINING_SAMPLES = 200

clean_data_file = '../../Data/conv-clean.txt'

data = None
data_string = ''

with open(clean_data_file, mode='r') as f:
    data_string = f.read().strip()
    data = data_string.split('\n')[:NUMBER_OF_TRAINING_SAMPLES]

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

for i, sen in enumerate(x1):
    for j, token in enumerate(word_tokenize(sen)):
        X[i, j, :] = w2v[token] if w2v.vocab.has_key(token) else w2v['unk']

for i, sen in enumerate(y1):
    for j, token in enumerate(word_tokenize(sen)):
        if w2v.vocab.has_key(token):
            Y[i, j, word_to_index[token]] = 1
        else:
            Y[i, j, word_to_index['unk']] = 1

model = SimpleSeq2seq(input_dim=vocab_dim, hidden_dim=300, output_length=max_length, output_dim=len(vocab), depth=2)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X, Y, nb_epoch=10)
