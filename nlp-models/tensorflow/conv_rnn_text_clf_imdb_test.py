from conv_rnn_text_clf import ConvLSTMClassifier
import tensorflow as tf
import numpy as np


max_seq_len = 250
vocab_size = 20000
batch_size = 32


def sort_by_len(x, y):
    idx = sorted(range(len(x)), key=lambda i: len(x[i]))
    return x[idx], y[idx]


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

    X_train, y_train = sort_by_len(X_train, y_train)
    X_test, y_test = sort_by_len(X_test, y_test)

    clf = ConvLSTMClassifier(max_seq_len, vocab_size)
    log = clf.fit(X_train, y_train, batch_size=32, n_epoch=2, keep_prob=0.8, val_data=(X_test,y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
