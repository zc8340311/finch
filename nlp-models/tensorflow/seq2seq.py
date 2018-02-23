import tensorflow as tf
import numpy as np


class Seq2Seq:
    def __init__(self, rnn_size, n_layers, X_word2idx, encoder_embedding_dim, Y_word2idx, decoder_embedding_dim,
                 sess=tf.Session(), grad_clip=5.0):
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.X_word2idx = X_word2idx
        self.encoder_embedding_dim = encoder_embedding_dim
        self.Y_word2idx = Y_word2idx
        self.decoder_embedding_dim = decoder_embedding_dim
        self.sess = sess
        self.register_symbols()
        self.build_graph()
    # end constructor


    def build_graph(self):
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_decoder_layer()
        self.add_backward_path()
    # end method build_graph


    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.shape(self.X)[0]
    # end method add_input_layer


    def lstm_cell(self, reuse=False):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    # end method lstm_cell


    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.X_word2idx), self.encoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))            
        _, self.encoder_state = tf.nn.dynamic_rnn(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]), 
            inputs = tf.nn.embedding_lookup(encoder_embedding, self.X),
            sequence_length = self.X_seq_len,
            dtype = tf.float32)
        self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.n_layers))
    # end method add_encoder_layer
    

    def processed_decoder_input(self):
        #main = tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
        main = self.Y[:, :-1] # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._y_go), main], 1)
        return decoder_input
    # end method add_decoder_layer


    def add_decoder_layer(self):
        with tf.variable_scope('decode'):
            decoder_embedding = tf.get_variable('decoder_embedding', [len(self.Y_word2idx), self.decoder_embedding_dim],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
                sequence_length = self.Y_seq_len,
                time_major = False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.n_layers)]),
                helper = training_helper,
                initial_state = self.encoder_state,
                output_layer = tf.layers.Dense(len(self.Y_word2idx)))
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output
        
        with tf.variable_scope('decode', reuse=True):
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = tf.get_variable('decoder_embedding'),
                start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
                end_token = self._y_eos)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(reuse=True) for _ in range(self.n_layers)]),
                helper = predicting_helper,
                initial_state = self.encoder_state,
                output_layer = tf.layers.Dense(len(self.Y_word2idx), _reuse=True))
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.sample_id
    # end method add_decoder_layer


    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)

        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))
    # end method add_backward_path


    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens
    # end method pad_sentence_batch


    def next_batch(self, X, Y, batch_size):
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, self._y_pad)
            yield (np.array(padded_X_batch),
                   np.array(padded_Y_batch),
                   X_batch_lens,
                   Y_batch_lens)
    # end method next_batch


    def fit(self, X_train, Y_train, val_data, n_epoch=60, display_step=50, batch_size=128):
        X_test, Y_test = val_data
        X_test_batch, Y_test_batch, X_test_batch_lens, Y_test_batch_lens = next(
        self.next_batch(X_test, Y_test, batch_size))

        self.sess.run(tf.global_variables_initializer())
        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                self.next_batch(X_train, Y_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X: X_train_batch,
                                                                     self.Y: Y_train_batch,
                                                                     self.X_seq_len: X_train_batch_lens,
                                                                     self.Y_seq_len: Y_train_batch_lens})
                if local_step % display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.X: X_test_batch,
                                                         self.Y: Y_test_batch,
                                                         self.X_seq_len: X_test_batch_lens,
                                                         self.Y_seq_len: Y_test_batch_lens})
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | test_loss: %.3f"
                        % (epoch, n_epoch, local_step, len(X_train)//batch_size, loss, val_loss))
    # end method fit


    def infer(self, input_word, X_idx2word, Y_idx2word, batch_size=128):        
        input_indices = [self.X_word2idx.get(char, self._x_unk) for char in input_word]
        out_indices = self.sess.run(self.predicting_ids, {
            self.X: [input_indices] * batch_size,
            self.X_seq_len: [len(input_indices)] * batch_size})[0]
        
        print('\nSource')
        print('Word: {}'.format([i for i in input_indices]))
        print('IN: {}'.format(' '.join([X_idx2word[i] for i in input_indices])))
        
        print('\nTarget')
        print('Word: {}'.format([i for i in out_indices]))
        print('OUT: {}'.format(' '.join([Y_idx2word[i] for i in out_indices])))
    # end method infer


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']

        self._y_go = self.Y_word2idx['<GO>']
        self._y_eos = self.Y_word2idx['<EOS>']
        self._y_pad = self.Y_word2idx['<PAD>']
        self._y_unk = self.Y_word2idx['<UNK>']
    # end method add_symbols
# end class