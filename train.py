import numpy as np
import tensorflow as tf
from utils import *
from model import *


# load data
train_x = np.load('x.npy')
train_y = np.load('y.npy')
dictionary = load_pickle('dictionary.pkl')

# Parameters
learning_rate = 1e-3
epochs = 100
batch_size = 20

model = Generator(input_dim=train_x.shape[1], 
                  embedding_size=200, 
                  time_step=train_y.shape[1], 
                  hidden_dim=200, 
                  vocab=dictionary)

# training interface
with tf.Session() as sess:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=40)
    sess.run(init)
    for epoch in range(epochs):
        cost = 0.0
        for itr in range((train_x.shape[0] - 1) // batch_size + 1):
            _, loss = sess.run([optimizer, model.loss], 
                               feed_dict={model.features: train_x, model.titles: train_y})
            cost += loss
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.6f}".format(cost))
        saver.save(sess,'./model_ckpt/model',global_step=epoch+1)

    print("Optimization Finished!")

    pred = sess.run(model.pred, feed_dict={model.features: train_x})
    print(convert(pred, dictionary))
