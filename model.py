import tensorflow as tf


def linear_layer(input, output_dim, name, activation=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_dim = input.shape[1]
        weights = tf.get_variable("weight", [input_dim, output_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("bias", [output_dim],
            initializer=tf.zeros_initializer())
        output = tf.matmul(input, weights) + biases

        if activation != None:
            output = activation(output)

        return output

def embedding_layer(word_ids, vocabulary_size, embedding_size, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, embedding_size],
                                          initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)

        return embedded_word_ids


class Generator(object):
    def __init__(self, input_dim, embedding_size, time_step, hidden_dim, vocab):
        # input placeholder
        self.features = tf.placeholder(tf.float32, shape=(None, input_dim))
        self.titles = tf.placeholder(tf.int32, shape=(None, time_step))
        batch_size = tf.shape(self.features)[0]
        vocab_size = len(vocab)

        # encoder
        h1 = linear_layer(self.features, 800, activation=tf.nn.sigmoid, name='l1')
        z = linear_layer(h1, 400, activation=tf.nn.sigmoid, name='l2')

        # decoder
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, activation=tf.nn.relu, initializer=tf.orthogonal_initializer())
        c = linear_layer(z, hidden_dim, name='map_c')
        h = linear_layer(z, hidden_dim, name='map_h')


        loss = 0.0
        pred = []
        # assert
        assert vocab['<PAD>'] == 0 and vocab['<EOS>'] == 1
        # first input <EOS>
        next_input = embedding_layer(tf.ones([batch_size], tf.int32), vocab_size, 
                                     embedding_size, name='word_embedding')
        # initial state (LSTMStateTuple)
        state = (c, h)
        for t in range(time_step):
            output, state = lstm_cell(next_input, state)
            logits = linear_layer(output, vocab_size, name='output', reuse=(t != 0))
            pred.append(tf.argmax(logits, 1))
            next_input = embedding_layer(pred[-1], vocab_size, embedding_size, name='word_embedding', reuse=True)
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.titles[:, t]))

        self.loss = loss
        self.pred = tf.stack(pred, axis=1)
