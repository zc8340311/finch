from rnn_attn_estimator_imdb_config import args
import tensorflow as tf


def forward_pass(x, reuse):
    batch_size = tf.shape(x)[0]
    seq_len = tf.count_nonzero(x, 1, dtype=tf.int32)

    with tf.variable_scope('word_embedding', reuse=reuse) as scope:
        embedded = tf.contrib.layers.embed_sequence(
            x, args.vocab_size, args.embedding_dims, scope=scope, reuse=reuse)
        embedded = tf.layers.dropout(embedded, args.dropout_rate, training=(not reuse))

    with tf.variable_scope('rnn', reuse=reuse):
        rnn_out, final_state = tf.nn.dynamic_rnn(
            rnn_cell(reuse=reuse), embedded, sequence_length=seq_len, dtype=tf.float32)

    with tf.variable_scope('attention', reuse=reuse):
        v = tf.get_variable("attention_v", [args.attn_size], tf.float32)
        query = tf.layers.dense(tf.expand_dims(final_state.h, 1), args.attn_size) # (B, 1, D)
        keys = tf.layers.dense(rnn_out, args.attn_size)                           # (B, T, D)
        align = tf.reduce_sum(v * tf.tanh(keys + query), [2])
        align = _softmax(align)
        logits = tf.squeeze(tf.matmul(tf.transpose(rnn_out, [0,2,1]), tf.expand_dims(align, 2)), 2)
    
    with tf.variable_scope('output_layer', reuse=reuse):
        logits = tf.layers.dense(logits, args.num_classes, reuse=reuse)
    return logits
# end function


def model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return _model_fn_train(features, labels, mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        return _model_fn_eval(features, labels, mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return _model_fn_predict(features, mode)
# end function


def _model_fn_train(features, labels, mode):
    logits = forward_pass(features['data'], reuse=False)
    
    with tf.name_scope('backward_pass'):
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
        acc_op = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(logits,1), labels), tf.float32))

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('acc', acc_op)
        summary_hook = tf.train.SummarySaverHook(
            save_steps=10, output_dir='./board/', summary_op=tf.summary.merge_all())
        tensor_hook = tf.train.LoggingTensorHook({'acc':acc_op}, every_n_iter=100)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        train_op = tf.train.AdamOptimizer().apply_gradients(
            zip(clipped_gradients, params), global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss_op,
        train_op = train_op,
        training_hooks = [tensor_hook, summary_hook])
# end function


def _model_fn_eval(features, labels, mode):
    logits = forward_pass(features['data'], reuse=False)
    predictions = tf.argmax(forward_pass(features['data'], reuse=True), axis=1)
    
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss_op,
        eval_metric_ops = {'test_acc': acc_op})
# end function


def _model_fn_predict(features, mode):
    logits = forward_pass(features['data'], reuse=False)
    predictions = tf.argmax(forward_pass(features['data'], reuse=True), axis=1)
    
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
# end function


def rnn_cell(reuse):
    cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.orthogonal_initializer(), reuse=reuse)
    return cell
# end function


def _softmax(tensor):
    exps = tf.exp(tensor)
    return exps / tf.reduce_sum(exps, 1, keep_dims=True)
# end method softmax