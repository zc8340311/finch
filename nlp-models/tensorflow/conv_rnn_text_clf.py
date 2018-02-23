import tensorflow as tf
import numpy as np
import math
import sklearn


class ConvLSTMClassifier:
    def __init__(self, max_seq_len, vocab_size, n_out=2, sess=tf.Session(),
                 embedding_dims=128, n_filters=64, kernel_size=5, pool_size=4, padding='valid',
                 cell_size=64):
        """
        Parameters:
        -----------
        seq_len: int
            Sequence length
        vocab_size: int
            Vocabulary size
        embedding_dims: int
            Word embedding dimensions
        n_filters: int
            Number output of filters in the convolution
        kernel_size: int
            Size of the 1D convolution window
        pool_size: int
            Size of the max pooling windows
        cell_size: int
            Number of units in the rnn cell
        n_out: int
            Output dimensions
        sess: object
            tf.Session() object 
        """
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.cell_size = cell_size
        self.n_out = n_out
        self.sess = sess
        self._pointer = None         # used to point to the forefront of neural network
        self.build_graph()
    # end constructor
 
 
    def build_graph(self):
        self.add_input_layer()
        self.add_word_embedding()
        self.add_conv1d(self.n_filters)
        self.add_pooling(self.pool_size)
        self.add_lstm_cells()
        self.add_dynamic_rnn()
        self.add_output_layer()   
        self.add_backward_path()
    # end method build_graph
 
 
    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.Y = tf.placeholder(tf.int64, [None])
        self.X_seq_lens = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X
    # end method add_input_layer


    def add_word_embedding(self):
        embedding = tf.get_variable('encoder', [self.vocab_size, self.embedding_dims], tf.float32,
                                     tf.random_uniform_initializer(-1.0, 1.0))
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        self._pointer = tf.nn.dropout(embedded, self.keep_prob)
    # end method add_word_embedding


    def add_conv1d(self, n_filters, strides=1):
        Y = tf.layers.conv1d(inputs = self._pointer,
                             filters = n_filters,
                             kernel_size  = self.kernel_size,
                             strides = strides,
                             padding = self.padding,
                             use_bias = True,
                             activation = tf.nn.relu)
        self._pointer = Y
    # end method add_conv1d


    def add_pooling(self, k):
        self._pointer = tf.layers.max_pooling1d(
            inputs = self._pointer,
            pool_size = k,
            strides = k,
            padding = self.padding)
    # end method add_maxpool


    def add_lstm_cells(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, initializer=tf.orthogonal_initializer())
    # end method add_rnn_cells


    def add_dynamic_rnn(self):      
        _, self._pointer = tf.nn.dynamic_rnn(self.cell, self._pointer,
                                            dtype = tf.float32,
                                            sequence_length = (self.X_seq_lens // self.pool_size),
                                            time_major = False)
    # end method add_dynamic_rnn


    def add_output_layer(self):
        self.logits = tf.layers.dense(self._pointer.h, self.n_out)
    # end method add_output_layer


    def add_backward_path(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.Y), tf.float32))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    # end method add_backward_path


    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, keep_prob=1.0, en_exp_decay=True):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0

        self.sess.run(tf.global_variables_initializer()) # initialize all variables
        for epoch in range(n_epoch):
            for local_step, ((X_batch, X_batch_lens), Y_batch) in enumerate(
                zip(self.next_batch(X, batch_size), self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(en_exp_decay, global_step, n_epoch, len(X), batch_size) 
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             {self.X: X_batch, self.Y: Y_batch,
                                              self.X_seq_lens: X_batch_lens,
                                              self.lr: lr,
                                              self.keep_prob: keep_prob})
                global_step += 1
                if local_step % 50 == 0:
                    print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, n_epoch, local_step, int(len(X)/batch_size), loss, acc, lr))

            if val_data is not None: # go through test dara, compute averaged validation loss and acc
                val_loss_list, val_acc_list = [], []
                for (X_test_batch, X_test_batch_lens), Y_test_batch in zip(self.next_batch(val_data[0], batch_size),
                                                                           self.gen_batch(val_data[1], batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],
                                                  {self.X: X_test_batch, self.Y: Y_test_batch,
                                                   self.X_seq_lens: X_test_batch_lens,
                                                   self.keep_prob: 1.0})
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)

            # append to log
            log['loss'].append(loss)
            log['acc'].append(acc)
            if val_data is not None:
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)
            # verbose
            if val_data is None:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                    "lr: %.4f" % (lr) )
            else:
                print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, n_epoch, loss, acc),
                    "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                    "lr: %.4f" % (lr) )
        # end "for epoch in range(n_epoch):"

        return log
    # end method fit


    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for (X_test_batch, X_test_batch_lens) in self.next_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits, {self.X: X_test_batch,
                                                     self.X_seq_lens: X_test_batch_lens,
                                                     self.keep_prob: 1.0})
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)
    # end method predict


    def pad_sentence_batch(self, sentence_batch, pad_int=0):
        padded_seqs = []
        seq_lens = []
        for sentence in sentence_batch:
            if len(sentence) < self.max_seq_len:
                padded_seqs.append(sentence + [pad_int] * (self.max_seq_len - len(sentence)))
                seq_lens.append(len(sentence))
            else:
                padded_seqs.append(sentence[:self.max_seq_len])
                seq_lens.append(self.max_seq_len)
        return padded_seqs, seq_lens
    # end method pad_sentence_batc


    def next_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            padded_seqs, seq_lens = self.pad_sentence_batch(arr[i : i+batch_size])
            yield padded_seqs, seq_lens
    # end method gen_batch


    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]
    # end method gen_batch


    def decrease_lr(self, en_exp_decay, global_step, n_epoch, len_X, batch_size):
        if en_exp_decay:
            max_lr = 0.005
            min_lr = 0.001
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