import numpy as np
import random
import tensorflow as tf
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from itertools import combinations, combinations_with_replacement
import torch

import os
from sklearn.model_selection import train_test_split
import sys
from triplet.different_triplet_loss.config import *

# @tf.function
import tensorflow as tf

# supervised contrastive loss
def triplet_loss_bh(y_true, y_pred):
    def body(i, loss):
        indices = tf.range(i, n, delta=P * K, dtype=tf.int32)
        zi = tf.gather(Zi, indices)
        za = tf.gather(Za, indices)
        D_ia = tf.reduce_sum(zi * za, axis=1)
        mask = tf.gather(y_true[:, 0], indices)
        mask = tf.not_equal(mask, 0)
        D_ip = tf.boolean_mask(D_ia, mask)
        loss += tf.reduce_sum((-1 / K) * tf.math.log(tf.math.exp(D_ip / tau) / tf.reduce_sum(tf.math.exp(D_ia / tau))))
        return i + 1, loss

    def condition(i, loss):
        return i < K*P

    Zi, Za = y_pred[:, :emb_size], y_pred[:, -emb_size:]  # Zi: embedding of real, Za: embedding of synthetic

    tau = 100
    # K, P = 2, 2
    n = (K * P)**2

    
    
    i = tf.constant(0)
    initial_loss = tf.constant(0.0, dtype=tf.float32)
    _, loss = tf.while_loop(condition, body, loop_vars=[i, initial_loss], parallel_iterations=1)

    return loss / (K*P)


# K classes, P samples per class for each batch
class DataGeneratorBH_realAnchor(tf.keras.utils.Sequence):
    'Generates data for Keras'
    
    # generate batch with P classes and K samples per class
    
    def __init__(self, embed_model, emb_size=128, alpha=0.2, P=4, K=8, dim=(28, 52), class_num=5,
                 shuffle=True, data_real=None, labels_real=None, data_syn=None, labels_syn=None):
        """

        Args:
            P: randomly select P classes per batch
            K: randomly select K samples per class per batch. batch size: P*K
            dim: dimension of anchor/positive/negative
            shuffle:
        """
        self.dim = dim
        self.class_num = class_num
        self.P = P
        self.K = K
        self.emb_size = emb_size
        self.embed_model = embed_model
        self.emb_size = 128
        self.alpha = alpha
        self.data_real = data_real
        self.data_syn = data_syn
        self.labels_real = labels_real
        self.labels_syn = labels_syn
        self.semi_cnt = 0
        self.list_IDs_real = np.arange(len(self.labels_real))
        self.list_IDs_syn = np.arange(len(self.labels_syn))
        self.real_cnt = np.zeros(class_num).astype(int)
        self.syn_cnt = np.zeros(class_num).astype(int)
        self.shuffle = shuffle
        
        self.indexes_real = np.arange(len(self.list_IDs_real))
        self.indexes_syn = np.arange(len(self.list_IDs_syn))
        self.start = True
        # self.on_epoch_end()
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels_syn) / (self.P * self.K)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # real
        indexes = []
        y = np.ones((self.P * self.K * 2, self.emb_size))  # y0 = label; real data, y1 = 1; synthetic data, y1 = 0
        
        classes_tmp = random.sample(list(np.arange(self.class_num)), self.P)
        # cnt = 0
        labels_real_temp = self.labels_real[self.indexes_real]
        for l in classes_tmp:
            class_index = (labels_real_temp == l)
            # sample_index = random.sample(self.indexes[class_index], self.K)
            if self.real_cnt[l] + self.K >= len(self.indexes_real[class_index]):
                sample_index = self.indexes_real[class_index][-self.K:]
            else:
                sample_index = self.indexes_real[class_index][self.real_cnt[l]:self.real_cnt[l] + self.K]
            self.real_cnt[l] = (self.real_cnt[l] + self.K) % len(class_index)
            if len(sample_index) < self.K:
                sample_index = np.concatenate((sample_index, [sample_index[0]] * (self.K - len(sample_index))))
            # y[cnt, 0] = l
            # cnt += 1
            indexes.extend(sample_index)
        
        list_IDs_temp = [self.list_IDs_real[i] for i in indexes]
        
        # X_real = self.data_real[list_IDs_temp][:, :, :, np.newaxis]
        X_real = self.data_real[list_IDs_temp]
        y[:self.K * self.P, 0] = self.labels_real[list_IDs_temp]  # y0 = label
        y[:self.K * self.P, 1] = np.zeros(self.P * self.K)
        
        # synthetic
        indexes = []
        labels_syn_temp = self.labels_syn[self.indexes_syn]
        for l in classes_tmp:
            class_index = (labels_syn_temp == l)
            # sample_index = random.sample(self.indexes[class_index], self.K)
            if self.syn_cnt[l] + self.K >= len(self.indexes_syn[class_index]):
                sample_index = self.indexes_syn[class_index][-self.K:]
            else:
                sample_index = self.indexes_syn[class_index][self.syn_cnt[l]:self.syn_cnt[l] + self.K]
            self.syn_cnt[l] = (self.syn_cnt[l] + self.K) % len(class_index)
            if len(sample_index) < self.K:
                sample_index = np.concatenate((sample_index, [sample_index[0]] * (self.K - len(sample_index))))
           
            indexes.extend(sample_index)
        
        list_IDs_temp = [self.list_IDs_syn[i] for i in indexes]
        
        # X_syn = self.data_syn[list_IDs_temp][:, :, :, np.newaxis]
        X_syn = self.data_syn[list_IDs_temp]
        y[self.K * self.P:, 0] = self.labels_syn[list_IDs_temp]  # y0 = label
        y[self.K * self.P:, 1] = np.zeros(self.P * self.K)
        
        batch_size = self.P * self.K
        
        label_real = y[:batch_size, 0]
        label_syn = y[batch_size:, 0]
        
        # all_pairs = np.array(list(combinations_with_replacement(range(len(label_real)), 2)))
        all_pairs = np.array([[i, j] for j in range(len(label_syn)) for i in range(len(label_real))])
        all_pairs = torch.LongTensor(all_pairs)
        # sorted_indices = tf.argsort(all_pairs[:, 0])
        # all_pairs = tf.gather(all_pairs, sorted_indices)
        positive_pairs = all_pairs[(label_real[all_pairs[:, 0]] == label_syn[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(label_real[all_pairs[:, 0]] != label_syn[all_pairs[:, 1]]).nonzero()]
        # negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]
        
        X = [np.vstack((X_real[positive_pairs[:, 0]], X_real[negative_pairs[:, 0]])), np.vstack((X_syn[positive_pairs[:, 1]], X_syn[negative_pairs[:, 1]]))]
        y = np.zeros((len(X[0]), emb_size))
        y[:len(positive_pairs[:, 0]), 0] = np.ones((len(positive_pairs[:, 0])))     # y[:,0]=1 means positive pair, otherwise negative pair
        
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes_real)
            np.random.shuffle(self.indexes_syn)
        self.real_cnt = np.zeros(self.class_num).astype(int)
        self.syn_cnt = np.zeros(self.class_num).astype(int)
        self.start = False


# random batch
class DataGeneratorBH_realAnchor_randomBatch(tf.keras.utils.Sequence):
    'Generates data for Keras'
    
    # generate random batch with batch_size of real data and synthetic data
    
    def __init__(self, emb_size=128, alpha=0.2, batch_size=32, dim=(28, 52), class_num=5,
                 shuffle=True, data_real=None, labels_real=None, data_syn=None, labels_syn=None):
        
        self.dim = dim
        self.class_num = class_num
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.emb_size = 128
        self.alpha = alpha
        self.data_real = data_real
        self.data_syn = data_syn
        self.labels_real = labels_real
        self.labels_syn = labels_syn
        self.cnt_syn = 0
        self.cnt_real = 0
        self.list_IDs_real = np.arange(len(self.labels_real))
        self.list_IDs_syn = np.arange(len(self.labels_syn))
        self.indexes_real = np.arange(len(self.list_IDs_real))
        self.indexes_syn = np.arange(len(self.list_IDs_syn))
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels_syn) / (self.batch_size)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # real
        indexes = []
        y = np.ones((self.batch_size * 2, self.emb_size))  # y0 = label; real data, y1 = 1; synthetic data, y1 = 0
        
        if self.cnt_real + self.batch_size <= len(self.indexes_real):
            list_IDs_temp_real = [self.list_IDs_real[i] for i in
                                  self.indexes_real[self.cnt_real: self.cnt_real + self.batch_size]]
        else:
            index_real = self.indexes_real[self.cnt_real:]
            index_real = np.concatenate((index_real, [index_real[0]] * (self.batch_size - len(index_real))))
            list_IDs_temp_real = [self.list_IDs_real[i] for i in index_real]
        X_real = self.data_real[list_IDs_temp_real]
        y[:self.batch_size, 0] = self.labels_real[list_IDs_temp_real]  # y0 = label
        y[:self.batch_size, 1] = np.zeros(self.batch_size)
        
        # synthetic
        if self.cnt_syn + self.batch_size <= len(self.indexes_syn):
            list_IDs_temp_syn = [self.list_IDs_syn[i] for i in
                                 self.indexes_syn[self.cnt_syn: self.cnt_syn + self.batch_size]]
        else:
            index_syn = self.indexes_syn[self.cnt_syn:]
            index_syn = np.concatenate((index_syn, [index_syn[0]] * (self.batch_size - len(index_syn))))
            list_IDs_temp_syn = [self.list_IDs_syn[i] for i in index_syn]
        X_syn = self.data_syn[list_IDs_temp_syn]
        y[self.batch_size:, 0] = self.labels_syn[list_IDs_temp_syn]  # y0 = label
        y[self.batch_size:, 1] = np.zeros(self.batch_size)
        
        label_real = y[:self.batch_size, 0]
        label_syn = y[self.batch_size:, 0]
        
        # all_pairs = np.array(list(combinations_with_replacement(range(len(label_real)), 2)))
        all_pairs = np.array([[i, j] for j in range(len(label_syn)) for i in range(len(label_real))])
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(label_real[all_pairs[:, 0]] == label_syn[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(label_real[all_pairs[:, 0]] != label_syn[all_pairs[:, 1]]).nonzero()]
        # balance
        # negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]
        
        X = [np.vstack((X_real[positive_pairs[:, 0]], X_real[negative_pairs[:, 0]])),
             np.vstack((X_syn[positive_pairs[:, 1]], X_syn[negative_pairs[:, 1]]))]
        y = np.zeros((len(X[0]), emb_size))
        y[:len(positive_pairs[:, 0]), 0] = np.ones((len(positive_pairs[:, 0])))  # y[:,0]=1 means positive pair, otherwise negative pair
       
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes_real)
            np.random.shuffle(self.indexes_syn)
        self.cnt_real = 0
        self.cnt_syn = 0


if __name__ == "__main__":
    t1 = tf.constant([[1, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [0, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                      [0, 2, 3, 4]], tf.float32)
    t2 = tf.constant(
        [[1, 2, 3, 40], [12, 2, 3, 4], [1, 21, 3, 4], [1, 2, 35, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
         [1, 2, 3, 4], [1, 2, 3, 40], [12, 2, 3, 4], [1, 21, 3, 4], [1, 2, 35, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
         [1, 2, 3, 4]], tf.float32)
    triplet_loss_bh(t1, t2)