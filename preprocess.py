import os
import pysrt
import pandas as pd
import numpy as np
from collections import Counter
from utils import *

folder_path = 'scripts'
movies = pd.read_csv('list.csv').values
feature_dim = 10000

corpus = []
for ch, en in movies:
    movie_folder_path = os.path.join(folder_path, en)
    dirs = os.listdir(movie_folder_path)
    for file in dirs:
        if os.path.splitext(file)[-1] == '.srt':
            movie_srt_path = os.path.join(movie_folder_path, file)
            subs = pysrt.open(movie_srt_path)
            context = []
            for sub in subs:
                context += tokenize(sub.text)
            corpus.append((ch, context))

counter = Counter()
for _, script in corpus:
    counter.update(script)
vocab = [word for word, cnt in counter.most_common()]
# print(len(vocab))
if len(vocab) > feature_dim:
    vocab = vocab[-feature_dim:]

# bag of words
matrix = np.empty((len(corpus), len(vocab)))
for i, (_, script) in enumerate(corpus):
    counter = Counter(script)
    vec = np.array([counter[word] for word in vocab])
    matrix[i] = vec / sum(vec)

max_len = 0
counter = Counter()
for title, _ in corpus:
    line = tokenize(title)
    max_len = max(max_len, len(line))
    counter.update(line)

dictionary = {u'<PAD>': 0, u'<EOS>': 1}
dictionary.update({
    word: len(dictionary) + i for i, word in enumerate(counter.keys())})
# print(len(dictionary))

max_len += 1
target = np.empty((len(corpus), max_len), int)
for i, (title, _) in enumerate(corpus):
    vec = np.zeros(max_len)
    line = tokenize(title) + ['<EOS>']
    for idx, word in enumerate(line):
        vec[idx] = dictionary[word]
    target[i] = vec

np.save('x.npy', matrix)
np.save('y.npy', target)
save_pickle(dictionary, 'dictionary.pkl')
save_pickle(vocab, 'vocab.pkl')
