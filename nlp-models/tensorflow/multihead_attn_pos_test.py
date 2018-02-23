import pos
from utils import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


SEQ_LEN = 10
BATCH_SIZE = 128
NUM_EPOCH = 1


def forward_pass(sources, params, reuse=False,
                 dropout_rate=0.2, hidden_units=128, num_heads=4, num_blocks=1):
    with tf.variable_scope('forward_pass', reuse=reuse):
        en_masks = tf.sign(tf.abs(sources))     
        with tf.variable_scope('encoder_embedding', reuse=reuse):
            encoded = embed_seq(
                sources, params['vocab_size'], hidden_units, zero_pad=False, scale=True)
        with tf.variable_scope('encoder_positional_encoding', reuse=reuse):
            encoded += learned_positional_encoding(sources, hidden_units, zero_pad=False, scale=False)
        with tf.variable_scope('encoder_dropout', reuse=reuse):
            encoded = tf.layers.dropout(encoded, dropout_rate, training=(not reuse))
        for i in range(num_blocks):
            with tf.variable_scope('encoder_attn_%d'%i, reuse=reuse):
                encoded = multihead_attn(queries=encoded, keys=encoded, q_masks=en_masks, k_masks=en_masks,
                    num_units=hidden_units, num_heads=num_heads, dropout_rate=dropout_rate,
                    causality=False, reuse=reuse, activation=None)
            with tf.variable_scope('encoder_feedforward_%d'%i, reuse=reuse):
                encoded = pointwise_feedforward(encoded, num_units=[hidden_units, hidden_units],
                    activation=tf.nn.elu)
        return tf.layers.dense(encoded, params['n_class'])


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return _model_fn_train(features, labels, mode, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return _model_fn_predict(features, mode, params)


def _model_fn_train(features, labels, mode, params):
    logits = forward_pass(features['inputs'], params)
    _ = forward_pass(features['inputs'], params, reuse=True)

    loss_op = tf.contrib.seq2seq.sequence_loss(
        logits = logits,
        targets = labels,
        weights = tf.ones_like(labels, tf.float32))

    acc_op = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(logits,-1), labels), tf.float32))
    tf.summary.scalar('acc', acc_op)
    tensor_hook = tf.train.LoggingTensorHook({'acc':acc_op}, every_n_iter=100)
    
    train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op,
        training_hooks = [tensor_hook])


def _model_fn_predict(features, mode, params):
    _ = forward_pass(features['inputs'], params)
    logits = forward_pass(features['inputs'], params, reuse=True)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=tf.argmax(logits,-1))


def main():
    x_train, y_train, x_test, y_test, vocab_size, n_class, word2idx, tag2idx = pos.load_data()
    X_train, Y_train = to_train_seq(x_train, y_train)
    X_test, Y_test = to_test_seq(x_test, y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    params = {'vocab_size': vocab_size, 'n_class': n_class}
    tf_estimator = tf.estimator.Estimator(model_fn, params=params)

    tf_estimator.train(tf.estimator.inputs.numpy_input_fn(
            x = {'inputs':X_train}, y=Y_train,
            batch_size = BATCH_SIZE, num_epochs = NUM_EPOCH, shuffle = True))
    preds = tf_estimator.predict(tf.estimator.inputs.numpy_input_fn(
            x = {'inputs':X_test}, batch_size = BATCH_SIZE, shuffle = False))
    preds = np.array(list(preds))

    final_acc = (preds == Y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)

    sample = ['I', 'love', 'you']
    idx = np.atleast_2d([word2idx[w] for w in sample] + [0] * (SEQ_LEN - len(sample)))
    preds = tf_estimator.predict(tf.estimator.inputs.numpy_input_fn(
            x={'inputs':idx}, batch_size=1, shuffle=False))
    preds = np.array(list(preds))
    idx2tag = {idx : tag for tag, idx in tag2idx.items()}
    print(' '.join(sample))
    print(' '.join([idx2tag[idx] for idx in preds[0][:len(sample)]]))


def to_train_seq(*args):
    data = []
    for x in args:
        data.append(iter_seq(x))
    return data


def to_test_seq(*args):
    data = []
    for x in args:
        x = x[: (len(x) - len(x) % SEQ_LEN)]
        data.append(np.reshape(x, [-1, SEQ_LEN]))
    return data


def iter_seq(x, text_iter_step=1):
    return np.array([x[i : i+SEQ_LEN] for i in range(0, len(x)-SEQ_LEN, text_iter_step)])


if __name__ == '__main__':
    main()
