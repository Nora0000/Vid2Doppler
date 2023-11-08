import numpy as np
import random
import tensorflow as tf
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()


import os
from sklearn.model_selection import train_test_split
import sys
from config import *

def x_max(tensor, x):
	"""
return the x-th largest value in tensor
	Args:
		tensor: 2-d tensor
		x: int

	Returns:

	"""
	max_value = tf.reduce_max(tensor, axis=1)
	if x == 1:
		return max_value
	for i in range(x, 1, -1):
		tensor_with_max_replaced = tf.where(tf.equal(tensor, max_value), tf.constant(-float('inf'), dtype=tf.float32), tensor)
		next_max_value = tf.reduce_max(tensor_with_max_replaced)
	return next_max_value


def x_min(tensor, x):
	"""
return the x-th smallest value in tensor
	Args:
		tensor:
		x: int

	Returns:

	"""
	min_value = tf.reduce_min(tensor, axis=1)
	if x == 1:
		return min_value
	for i in range(x, 1, -1):
		tensor_with_max_replaced = tf.where(tf.equal(tensor, min_value), tf.constant(float('inf'), dtype=tf.float32), tensor)
		next_min_value = tf.reduce_min(tensor_with_max_replaced)
		min_value = next_min_value
	return next_min_value


# @tf.function
def triple_loss_bh(y_true, y_pred):
	alpha = 0.2
	num=8
	# num = batch_size * 2
	emb_syn = y_pred[ num // 2:, :]
	label_syn = y_true[num // 2:, 0]
	anchor = y_pred[: num // 2, :]
	label = y_true[: num // 2, 0]
	anchor = tf.cast(anchor, dtype=tf.float32)
	emb_syn = tf.cast(emb_syn, dtype=tf.float32)
	
	# Calculate Euclidean distances for each anchor-synthetic pair
	distances = tf.norm(anchor[:, np.newaxis, :] - emb_syn, axis=2)
	
	# Create a mask for the positive and negative pairs
	positive_mask = tf.cast(tf.equal(label[:, np.newaxis], label_syn), dtype=tf.float32)
	negative_mask = 1 - positive_mask
	
	# batch hard triplet loss
	D_ap = tf.reduce_max(distances*positive_mask, axis=1)
	D_an = tf.reduce_min(distances*negative_mask, axis=1)
	loss = tf.reduce_mean(tf.maximum(alpha + D_ap - D_an, 0))
	
	# batch x-th hard triplet
	# D_ap = x_max(distances*positive_mask, 4)
	# D_an = x_min(distances*negative_mask, 4)
	# loss = tf.reduce_mean(tf.maximum(alpha + D_ap - D_an, 0))
	
	# top-k hard triplet
	# D_ap, _ = tf.nn.top_k(distances*positive_mask, k=3)
	# D_an, _ = tf.nn.top_k(-distances*negative_mask, k=3)
	# D_an = -D_an
	# loss = tf.reduce_mean(tf.maximum(alpha + D_ap[:, np.newaxis, :] - D_an, 0))
	
	
	# batch all triplet loss
	# D_ap = distances * positive_mask
	# D_an = distances * negative_mask
	# loss = tf.reduce_mean(tf.maximum(alpha + D_ap[:, np.newaxis, :] - D_an, 0))
	
	# half positive and negative
	# D_ap = distances*positive_mask
	# D_an = distances*negative_mask
	# D_ap = -tf.sort(-D_ap)
	# D_an = tf.sort(D_an)      # 需要去掉是0的，那些是positive对应的0位置的无效值
	# loss = tf.maximum(tf.reduce_mean(D_ap[:K//2]) - tf.reduce_mean(D_an[:K*(P-1)//2]) + alpha, 0)
	
	# the middle percentage of positives and negatives
	# D_ap = distances * positive_mask
	# D_an = distances * negative_mask
	# D_ap = -tf.sort(-D_ap)
	# D_an = tf.sort(D_an)
	# lp = int(tf.reduce_min(tf.reduce_sum(positive_mask, axis=1)))
	# ln = int(tf.reduce_min(tf.reduce_sum(negative_mask, axis=1)))
	# # discard 0-values which are masked
	# D_ap = D_ap[:, :lp]
	# D_an = D_an[:, -ln:]
	# start_n = tf.dtypes.cast((float(ln) - 1) * 0.1, tf.int32)
	# end_n = tf.dtypes.cast((float(ln) - 1) * 0.9, tf.int32)
	# loss = tf.reduce_mean(tf.maximum(alpha + D_ap[:, np.newaxis, :] - D_an[:, start_n:end_n, np.newaxis], 0))
	# loss = tf.reduce_mean(tf.maximum(alpha + D_ap[:, np.newaxis, :] - D_an[:, :, np.newaxis], 0))
	
	# supervised contrastive learning loss
	# positive_mask = tf.cast(tf.equal(label[:, np.newaxis], label_syn), tf.float32)
	# negative_mask = 1 - positive_mask
	# # pnum = tf.reduce_sum(tf.cast(positive_mask, tf.float32), axis=1)[0]
	# emb_p = positive_mask[:, np.newaxis, :] * tf.transpose(emb_syn)
	# # emb_p = tf.gather(emb_p, tf.argsort(emb_p[:, 0])[-K:])
	# D_ap = tf.reduce_sum(anchor[:, np.newaxis, :] * tf.transpose(emb_p, [0, 2, 1]), axis=2)
	# D_aa = tf.reduce_sum(anchor[:, np.newaxis, :] * emb_syn, axis=2)
	# tau = 1
	# Sum_aa = tf.reduce_sum(tf.exp(D_aa/tau), axis=1)
	# loss = tf.reduce_mean(-tf.math.log(tf.exp(D_ap/tau) / Sum_aa[:, np.newaxis]))
	
	# softplus
	# loss = tf.reduce_sum(tf.math.log(1 + tf.math.exp(alpha + D_ap - D_an)))
	
	return loss
		

class DataGeneratorBH_realAnchor(tf.keras.utils.Sequence):
	'Generates data for Keras'
	# generate batch with P classes and K samples per class
	
	def __init__(self, emb_size=128, alpha=0.2, P=4, K=8, dim=(28, 52), class_num=5,
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
		self.on_epoch_end()
		
	
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.labels_syn) / (self.P * self.K)))
	
	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		# real
		indexes = []
		y = np.ones((self.P * self.K * 2, self.emb_size))	# y0 = label; real data, y1 = 1; synthetic data, y1 = 0
		
		classes_tmp = random.sample(list(np.arange(self.class_num)), self.P)
		# cnt = 0
		labels_real_temp = self.labels_real[self.indexes_real]
		for l in classes_tmp:
			class_index = (labels_real_temp==l)
			# sample_index = random.sample(self.indexes[class_index], self.K)
			if self.real_cnt[l] + self.K >= len(self.indexes_real[class_index]):
				sample_index = self.indexes_real[class_index][-self.K:]
			else:
				sample_index = self.indexes_real[class_index][self.real_cnt[l]:self.real_cnt[l] + self.K]
			self.real_cnt[l]  = (self.real_cnt[l] + self.K) % len(class_index)
			if len(sample_index) < self.K:
				sample_index = np.concatenate((sample_index, [sample_index[0]]*(self.K - len(sample_index))))
			# y[cnt, 0] = l
			# cnt += 1
			indexes.extend(sample_index)
			
		list_IDs_temp = [self.list_IDs_real[i] for i in indexes]
		
		X = self.data_real[list_IDs_temp]
		y[:self.K*self.P, 0] = self.labels_real[list_IDs_temp]  # y0 = label
		y[:self.K*self.P, 1] = np.zeros(self.P * self.K)
		
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
				sample_index = np.concatenate((sample_index, [sample_index[0]]*(self.K - len(sample_index))))
			# y[cnt, 0] = l
			# y[cnt, 1] = 0
			# cnt += 1
			indexes.extend(sample_index)
			
		list_IDs_temp = [self.list_IDs_syn[i] for i in indexes]
		
		X = np.vstack((X, self.data_syn[list_IDs_temp]))
		y[self.K*self.P:, 0] = self.labels_syn[list_IDs_temp]  # y0 = label
		y[self.K*self.P:, 1] = np.zeros(self.P * self.K)
		X = X[:, :, :, np.newaxis]
		
		return X, y
	
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes_real = np.arange(len(self.list_IDs_real))
		self.indexes_syn = np.arange(len(self.list_IDs_syn))
		if self.shuffle == True:
			np.random.shuffle(self.indexes_real)
			np.random.shuffle(self.indexes_syn)
		self.real_cnt = np.zeros(self.class_num).astype(int)
		self.syn_cnt = np.zeros(self.class_num).astype(int)


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
		
		if self.cnt_real+self.batch_size <= len(self.indexes_real):
			list_IDs_temp_real = [self.list_IDs_real[i] for i in self.indexes_real[self.cnt_real: self.cnt_real+self.batch_size]]
		else:
			index_real = self.indexes_real[self.cnt_real:]
			index_real = np.concatenate((index_real, [index_real[0]]*(self.batch_size - len(index_real))))
			list_IDs_temp_real = [self.list_IDs_real[i] for i in index_real]
		X_real = self.data_real[list_IDs_temp_real]
		y[:self.batch_size, 0] = self.labels_real[list_IDs_temp_real]  # y0 = label
		y[:self.batch_size, 1] = np.zeros(self.batch_size)
		
		
		# synthetic
		if self.cnt_syn+self.batch_size <= len(self.indexes_syn):
			list_IDs_temp_syn = [self.list_IDs_syn[i] for i in self.indexes_syn[self.cnt_syn: self.cnt_syn+self.batch_size]]
		else:
			index_syn= self.indexes_syn[self.cnt_syn:]
			index_syn = np.concatenate((index_syn, [index_syn[0]]*(self.batch_size - len(index_syn))))
			list_IDs_temp_syn = [self.list_IDs_syn[i] for i in index_syn]
		X_syn = self.data_syn[list_IDs_temp_syn]
		y[:self.batch_size, 0] = self.labels_syn[list_IDs_temp_syn]  # y0 = label
		y[:self.batch_size, 1] = np.zeros(self.batch_size)
		
		X = np.vstack((X_real, X_syn))
		y[self.batch_size:, 0] = self.labels_syn[list_IDs_temp_syn]  # y0 = label
		y[self.batch_size:, 1] = np.zeros(self.batch_size)
		X = X[:, :, :, np.newaxis]
		
		return X, y
	
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		
		if self.shuffle == True:
			np.random.shuffle(self.indexes_real)
			np.random.shuffle(self.indexes_syn)
		self.cnt_real = 0
		self.cnt_syn = 0


if __name__ == "__main__":
	t1 = tf.constant([[1,2,3,4], [2,2,3,4], [1,2,3,4], [2,2,3,4], [2,2,3,4], [1,2,3,4],[1,2,3,4], [2,2,3,4]])
	t2 = tf.constant([[1,2,3,40], [12,2,3,4], [1,21,3,4], [1,2,35,4], [1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]])
	triple_loss_bh(t1/10, t2/10)