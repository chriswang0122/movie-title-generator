import os
import pysrt
import numpy as np
import tensorflow as tf
from collections import Counter
from model import *
from utils import *


dictionary = load_pickle('dictionary.pkl')
vocab = load_pickle('vocab.pkl')
train_x = np.load('x.npy')
train_y = np.load('y.npy')

test_path = 'test'
names = range(1, 11, 1)

corpus = []
for script in names:
    movie_srt_path = os.path.join(test_path, str(script) + '.srt')
    subs = pysrt.open(movie_srt_path)
    context = []
    for sub in subs:
        context += tokenize(sub.text)
    corpus.append(context)

matrix = np.empty((len(corpus), len(vocab)))
for i, script in enumerate(corpus):
    counter = Counter(script)
    vec = np.array([counter[word] for word in vocab])
    matrix[i] = vec / sum(vec)

model = Generator(input_dim=train_x.shape[1], 
                  embedding_size=200, 
                  time_step=train_y.shape[1], 
                  hidden_dim=200, 
                  vocab=dictionary)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "model_ckpt/model-100")
    pred = sess.run(model.pred, feed_dict={model.features: matrix})
    pred = convert(pred, dictionary)
    print(pred)

with open('results.txt', 'w') as f:
    for name, res in zip(names, pred):
        f.write(str(name) + '\t' + res + '\n')
