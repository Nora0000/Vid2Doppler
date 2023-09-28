import os.path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping


def plot_history(history, filename):
    # plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    loss = np.array(history['loss'])
    val_loss = np.array(history['val_loss'])
    # loss[loss > 10] = 10
    # val_loss[val_loss > 10] = 10
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training', 'Validation'])
    plt.grid()

    # plt.figure()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.plot(history['root_mean_squared_error'])
    # plt.plot(history['val_root_mean_squared_error'])
    # plt.legend(['Training', 'Validation'], loc='lower right')
    # plt.show()
    plt.savefig(filename)
    plt.clf()

emb_size = 128
alpha = 0.2
def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


class LossHistory(Callback):
    def __init__(self, log_file, loss_file):
        super().__init__()
        self.valLoss = None
        self.trainLoss = None
        self.logFile = log_file
        self.lossFile = loss_file

    def on_train_begin(self, logs={}):
        self.trainLoss = []
        self.valLoss = []

    def on_epoch_end(self, batch, logs={}):
        self.trainLoss.append(logs.get('loss'))
        self.valLoss.append(logs.get('val_loss'))

        np.save(self.lossFile, np.array([self.trainLoss, self.valLoss]))

        # plot loss curve
        plt.plot(self.trainLoss)
        plt.plot(self.valLoss)
        plt.legend(['Training', 'Validation'])
        plt.title("loss during training")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()
        plt.savefig(self.logFile)
        plt.clf()


class CustomizedEarlyStopping(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') <= alpha / 2:
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            print('Restoring model weights from the end of the best epoch.')
                        self.model.set_weights(self.best_weights)
