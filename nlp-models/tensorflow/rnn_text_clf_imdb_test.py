from __future__ import print_function
from rnn_text_clf import RNNTextClassifier
import tensorflow as tf
import numpy as np


vocab_size = 20000
batch_size = 32


def sort_by_len(x, y):
    idx = sorted(range(len(x)), key=lambda i: len(x[i]))
    return x[idx], y[idx]


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

    X_train, y_train = sort_by_len(X_train, y_train)
    X_test, y_test = sort_by_len(X_test, y_test)
    
    clf = RNNTextClassifier(vocab_size, 2)
    log = clf.fit(X_train, y_train, n_epoch=2, batch_size=batch_size, keep_prob=0.8, en_exp_decay=True,
                  val_data=(X_test, y_test))
    y_pred = clf.predict(X_test, batch_size)

    final_acc = (y_pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
