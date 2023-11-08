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
    emb_size = 128
    alpha = 0.2
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(positive_dist - negative_dist + alpha, 0.))


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
		y = np.ones((self.P * self.K, self.emb_size))  # y0 = label; real data, y1 = 1; synthetic data, y1 = 0
		
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
		label_real = self.labels_real[list_IDs_temp]  # y0 = label
		
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
			# y[cnt, 0] = l
			# y[cnt, 1] = 0
			# cnt += 1
			indexes.extend(sample_index)
		
		list_IDs_temp = [self.list_IDs_syn[i] for i in indexes]
		
		# X_syn = self.data_syn[list_IDs_temp][:, :, :, np.newaxis]
		X_syn = self.data_syn[list_IDs_temp]
		label_syn = self.labels_syn[list_IDs_temp]  # y0 = label
		
		batch_size = self.P * self.K
		
		# get embeddings, select hardest positive and negative for each anchor
		embeddings_real = self.embed_model.predict(X_real)
		embeddings_syn = self.embed_model.predict(X_syn)
		
		triplet_indexes = []
		for i in range(batch_size):
			l = label_real[i]
			# index_real = np.arange(len(label_real))[(label_real == l)]
			index_syn_pos = np.arange(len(label_syn))[(label_syn == l)]
			index_syn_neg = np.arange(len(label_syn))[(label_syn != l)]
			positive_dis = tf.reduce_mean(tf.square(embeddings_real[i] - embeddings_syn[index_syn_pos]), axis=1)
			neg_dis = tf.reduce_mean(tf.square(embeddings_real[i] - embeddings_syn[index_syn_neg]), axis=1)
			ap_hard = tf.argmax(positive_dis)
			an_hard = tf.argmin(neg_dis)
			if len(index_syn_pos) == 1:
				pos = index_syn_pos[0]
			else:
				pos = index_syn_pos[ap_hard.numpy()]
			if len(index_syn_neg) == 1:
				neg = index_syn_neg[0]
			else:
				neg = index_syn_neg[an_hard.numpy()]
			triplet_index = [i, pos, neg]
			if len(triplet_indexes) == 0:
				triplet_indexes = triplet_index
			else:
				triplet_indexes = np.vstack((triplet_indexes, triplet_index))
		X = [X_real[triplet_indexes[:, 0]], X_syn[triplet_indexes[:, 1]], X_syn[triplet_indexes[:, 2]]]
		
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
	
	def __init__(self, embed_model, emb_size=128, alpha=0.2, batch_size=32, dim=(28, 52), class_num=5,
	             shuffle=True, data_real=None, labels_real=None, data_syn=None, labels_syn=None):
		
		self.dim = dim
		self.class_num = class_num
		self.embed_model = embed_model
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
		y = np.ones((self.batch_size, self.emb_size))  # y0 = label; real data, y1 = 1; synthetic data, y1 = 0
		
		if self.cnt_real + self.batch_size <= len(self.indexes_real):
			list_IDs_temp_real = [self.list_IDs_real[i] for i in
			                      self.indexes_real[self.cnt_real: self.cnt_real + self.batch_size]]
		else:
			index_real = self.indexes_real[self.cnt_real:]
			index_real = np.concatenate((index_real, [index_real[0]] * (self.batch_size - len(index_real))))
			list_IDs_temp_real = [self.list_IDs_real[i] for i in index_real]
		X_real = self.data_real[list_IDs_temp_real]
		label_real = self.labels_real[list_IDs_temp_real]
		
		# synthetic
		if self.cnt_syn + self.batch_size <= len(self.indexes_syn):
			list_IDs_temp_syn = [self.list_IDs_syn[i] for i in
			                     self.indexes_syn[self.cnt_syn: self.cnt_syn + self.batch_size]]
		else:
			index_syn = self.indexes_syn[self.cnt_syn:]
			index_syn = np.concatenate((index_syn, [index_syn[0]] * (self.batch_size - len(index_syn))))
			list_IDs_temp_syn = [self.list_IDs_syn[i] for i in index_syn]
		X_syn = self.data_syn[list_IDs_temp_syn]
		label_syn = self.labels_syn[list_IDs_temp_syn]
		
		
		# get embeddings, select hardest positive and negative for each anchor
		embeddings_real = self.embed_model.predict(X_real)
		embeddings_syn = self.embed_model.predict(X_syn)

		
		triplet_indexes = []
		for i in range(self.batch_size):
			l = label_real[i]
			# index_real = np.arange(len(label_real))[(label_real == l)]
			index_syn_pos =  np.arange(len(label_syn))[(label_syn == l)]
			index_syn_neg = np.arange(len(label_syn))[(label_syn != l)]
			positive_dis = tf.reduce_mean(tf.square(embeddings_real[i]- embeddings_syn[index_syn_pos]), axis=1)
			neg_dis = tf.reduce_mean(tf.square(embeddings_real[i] - embeddings_syn[index_syn_neg]), axis=1)
			ap_hard = tf.argmax(positive_dis)
			an_hard = tf.argmin(neg_dis)
			if len(index_syn_pos) == 1:
				pos = index_syn_pos[0]
			else:
				pos = index_syn_pos[ap_hard.numpy()]
			if len(index_syn_neg) == 1:
				neg = index_syn_neg[0]
			else:
				neg = index_syn_neg[an_hard.numpy()]
			triplet_index = [i, pos, neg]
			if len(triplet_indexes) == 0:
				triplet_indexes = triplet_index
			else:
				triplet_indexes = np.vstack((triplet_indexes, triplet_index))
		X = [X_real[triplet_indexes[:, 0]], X_syn[triplet_indexes[:, 1]], X_syn[triplet_indexes[:, 2]]]
		
		return X, y
	
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		
		if self.shuffle == True:
			ii = np.arange(len(self.indexes_real))
			np.random.shuffle(ii)
			self.indexes_real = self.indexes_real[ii]
			self.indexes_syn = self.indexes_syn[ii]
			# np.random.shuffle(self.indexes_real)
			# np.random.shuffle(self.indexes_syn)
		self.cnt_real = 0
		self.cnt_syn = 0


if __name__ == "__main__":
	t1 = tf.constant([[1, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
	                  [0, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0, 2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4],
	                  [1, 2, 3, 4],
	                  [0, 2, 3, 4]], tf.float32)
	t2 = tf.constant(
		[[1, 2, 3, 40], [12, 2, 3, 4], [1, 21, 3, 4], [1, 2, 35, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
		 [1, 2, 3, 4], [1, 2, 3, 40], [12, 2, 3, 4], [1, 21, 3, 4], [1, 2, 35, 4], [1, 2, 3, 4], [1, 2, 3, 4],
		 [1, 2, 3, 4],
		 [1, 2, 3, 4]], tf.float32)
	triplet_loss_bh(t1, t2)