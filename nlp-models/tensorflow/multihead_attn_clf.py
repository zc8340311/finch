import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle
from utils import embed_seq, learned_positional_encoding, pointwise_feedforward, layer_norm


class Tagger:
    def __init__(self, vocab_size, n_out, seq_len,
                 dropout_rate=0.1, hidden_units=128, num_heads=8, num_blocks=1, sess=tf.Session()):
        self.vocab_size = vocab_size
        self.n_out = n_out
        self.seq_len = seq_len
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.sess = sess
        self._pointer = None
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_forward_path()
        self.add_crf_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_len])
        self.Y = tf.placeholder(tf.int32, [None, self.seq_len])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
    # end method add_input_layer


    def add_forward_path(self):
        with tf.variable_scope('encoder_embedding'):
            encoded = embed_seq(
                self.X, self.vocab_size, self.hidden_units, zero_pad=False, scale=True)
        with tf.variable_scope('encoder_positional_encoding'):
            encoded += learned_positional_encoding(self.X, self.hidden_units, zero_pad=False, scale=False)
        with tf.variable_scope('encoder_dropout'):
            encoded = tf.layers.dropout(encoded, self.dropout_rate, training=self.is_training)
        for i in range(self.num_blocks):
            with tf.variable_scope('restricted_attn_%d'%i):
                encoded = multihead_attn(queries=encoded, keys=encoded,
                    num_units=self.hidden_units, num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                    is_training=self.is_training, restricted=True)
            with tf.variable_scope('global_attn_%d'%i):
                encoded = multihead_attn(queries=encoded, keys=encoded,
                    num_units=self.hidden_units, num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                    is_training=self.is_training)
            with tf.variable_scope('encoder_feedforward_%d'%i):
                encoded = pointwise_feedforward(encoded, num_units=[4*self.hidden_units, self.hidden_units],
                    activation=tf.nn.elu)
        self.logits = tf.layers.dense(encoded, self.n_out)
    # end method add_forward_path


    def add_crf_layer(self):
        with tf.variable_scope('crf_loss'):
            self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs = self.logits,
                tag_indices = self.Y,
                sequence_lengths = self.X_seq_len)
        with tf.variable_scope('crf_loss', reuse=True):
            transition_params = tf.get_variable('transitions', [self.n_out, self.n_out])
        self.viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            self.logits, transition_params, self.X_seq_len)
    # end method add_crf_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.viterbi_sequence, self.Y), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X, Y, val_data, n_epoch=10, batch_size=128, en_exp_decay=True, en_shuffle=True):
        global_step = 0
        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch): # batch training
            if en_shuffle:
                X, Y = shuffle(X, Y)
                print("Data Shuffled")
            for local_step, (X_batch, Y_batch) in enumerate(zip(self.gen_batch(X, batch_size),
                                                                self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size)           
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X: X_batch, self.Y: Y_batch, self.lr: lr,
                                              self.X_seq_len: [X.shape[1]]*len(X_batch),
                                              self.is_training: True})
                global_step += 1
                if local_step % 50 == 0:
                    print ('Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f'
                           %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))
            # verbose
            print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                   "lr: %.4f" % (lr) )
            X_test, Y_test = val_data
            y_pred = self.predict(X_test, batch_size=batch_size)
            final_acc = (y_pred == Y_test).astype(np.float32).mean()
            print("final testing accuracy: %.4f" % final_acc)
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for X_test_batch in self.gen_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.viterbi_sequence,
                                      {self.X: X_test_batch,
                                       self.X_seq_len: len(X_test_batch)*[X_test.shape[1]],
                                       self.is_training: False})
            batch_pred_list.append(batch_pred)
        return np.vstack(batch_pred_list)
    # end method predict


    def infer(self, xs, x_len):
        viterbi_seq = self.sess.run(self.viterbi_sequence,
                                   {self.X: np.atleast_2d(xs),
                                    self.X_seq_len: np.atleast_1d(x_len),
                                    self.is_training: False})
        return np.squeeze(viterbi_seq,0)
    # end method infer


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.0005
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_X/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr
    # end method adjust_lr


    def list_avg(self, l):
        return sum(l) / len(l)
    # end method list_avg
# end class


def multihead_attn(queries, keys, num_units, num_heads, dropout_rate, is_training,
                   restricted=False):
    """
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q]
      keys: A 3d tensor with shape of [N, T_k, C_k]
    """
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]
    T_q = queries.get_shape().as_list()[1]                                         # max time length of query
    T_k = keys.get_shape().as_list()[1]                                            # max time length of key

    Q = tf.layers.dense(queries, num_units)                                        # (N, T_q, C)
    K = tf.layers.dense(keys, num_units)                                           # (N, T_k, C)
    V = tf.layers.dense(keys, num_units)                                           # (N, T_k, C)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         # (h*N, T_q, C/h) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                         # (h*N, T_k, C/h)

    align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                               # (h*N, T_q, T_k)
    align = align / (K_.get_shape().as_list()[-1] ** 0.5)                          # scale
    
    if restricted:
        paddings = tf.fill(tf.shape(align), float('-inf'))                         # exp(-large) -> 0
        lower_tri = tf.ones([T_q, T_k])                                            # (T_q, T_k)
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()     # (T_q, T_k)
        masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1])   # (h*N, T_q, T_k)
        align = tf.where(tf.equal(masks, 0), paddings, align)                      # (h*N, T_q, T_k)

    align = tf.nn.softmax(align)                                                   # (h*N, T_q, T_k)

    align = tf.layers.dropout(align, dropout_rate, training=is_training)           # (h*N, T_q, T_k)

    # Weighted sum
    outputs = tf.matmul(align, V_)                                                 # (h*N, T_q, C/h)
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)              # (N, T_q, C)
    # Residual connection
    outputs += queries                                                             # (N, T_q, C)   
    # Normalize
    outputs = layer_norm(outputs)                                                  # (N, T_q, C)
    return outputs