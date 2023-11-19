import os.path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping


# def triplet_loss(alpha=0.2, emb_size=128):
#     def loss(y_true, y_pred):
#         anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
#         positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
#         negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
#         return tf.maximum(positive_dist - negative_dist + alpha, 0.)
#     return loss


def triplet_loss(y_true, y_pred):
    emb_size = 128
    alpha = 0.2
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(positive_dist - negative_dist + alpha, 0.))


batch_size = 32
def triplet_loss_bh(y_true, y_pred):
    # use top half positive and negative
    emb_size = 128
    alpha = 0.2
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(-tf.sort(-tf.square(anchor - positive))[:batch_size//2])
    negative_dist = tf.reduce_mean(tf.sort(tf.square(anchor - negative))[:batch_size//2])
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


class AccuracyHistory(Callback):
    def __init__(self, log_file, acc_file):
        super().__init__()
        self.valAcc = None
        self.trainAcc = None
        self.logFile = log_file
        self.accFile = acc_file
    
    def on_train_begin(self, logs={}):
        self.trainAcc = []
        self.valAcc = []
    
    def on_epoch_end(self, batch, logs={}):
        self.trainAcc.append(logs.get('accuracy'))
        self.valAcc.append(logs.get('val_accuracy'))
        
        np.save(self.accFile, np.array([self.trainAcc, self.valAcc]))
        
        # plot loss curve
        plt.plot(self.trainAcc)
        plt.plot(self.valAcc)
        plt.legend(['Training', 'Validation'])
        plt.title("accuracy during training")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.grid()
        plt.savefig(self.logFile)
        plt.clf()

class CustomizedEarlyStopping(EarlyStopping):
    
    def on_epoch_end(self, epoch, logs=None):
        alpha = 0.2
        # if logs.get('val_loss') <= 0.1:
        if True:
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


class GradientExplosionCallback(Callback):
    def __init__(self, threshold=1.0):
        super(GradientExplosionCallback, self).__init__()
        self.threshold = threshold
        self.gradient_norm = []
    
    
    def on_epoch_end(self, epoch, logs=None):
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_variables)
        
        gradient_norms = []
        for var, gradient in zip(self.model.trainable_variables, gradients):
            gradient_norm = tf.norm(gradient)
            gradient_norms.append(gradient_norm)
        
        self.gradient_norm.append(gradient_norms)
        
        np.save("/home/mengjingliu/Vid2Doppler/Python/triplet/different_triplet_loss/gradient_norm.npy", np.array(self.gradient_norm))
        
        # # plot loss curve
        # plt.plot(self.trainLoss)
        # plt.plot(self.valLoss)
        # plt.legend(['Training', 'Validation'])
        # plt.title("loss during training")
        # plt.xlabel("epoch")
        # plt.ylabel("loss")
        # plt.grid()
        # plt.savefig(self.logFile)
        # plt.clf()