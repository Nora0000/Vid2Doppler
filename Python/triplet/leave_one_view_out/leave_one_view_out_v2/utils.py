import numpy as np
import tensorflow as tf
from config import *
import random

def end2end_loss(y_true, y_pred):
	
	alpha = 0.2
	anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:,
																						 2 * emb_size:3 * emb_size]
	positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
	negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
	triplet_loss = tf.reduce_mean(tf.maximum(positive_dist - negative_dist + alpha, 0.))
	
	beta1 = 2 * real_data_size
	beta2 = 1
	
	start = 3 * emb_size
	pre_a, pre_p, pre_n = y_pred[:, start:start + class_num], y_pred[:,
															  start + class_num:start + 2 * class_num], y_pred[:,
																										start + 2 * class_num:]
	label_a, label_p, label_n = y_true[:, start:start + class_num], y_true[:,
																	start + class_num:start + 2 * class_num], y_true[:,
																											  start + 2 * class_num:]
	epsilon = 1e-15
	pre_a = tf.clip_by_value(pre_a, epsilon, 1 - epsilon)
	pre_p = tf.clip_by_value(pre_p, epsilon, 1 - epsilon)
	pre_n = tf.clip_by_value(pre_n, epsilon, 1 - epsilon)
	cross_entropy_loss = tf.reduce_mean(- beta1 * tf.reduce_sum(label_a * tf.math.log(pre_a), axis=1) -
						  tf.reduce_sum(label_p * tf.math.log(pre_p), axis=1) - tf.reduce_sum(label_n * tf.math.log(pre_n), axis=1))

	return triplet_loss + beta2 * cross_entropy_loss


def tripletDataset_end2end(data_real, labels_real, data_syn, labels_syn):
	"""
	construct triplets: anchor (synthetic), positive (real data in the same class), negative (real data in different classes)
	balanced number of positives and negatives.
	input real and synthetic data are in the same order (aligned).
	
	include labels in Y to calculate cross entropy loss.
	
	Args:
		data_real:
		data_syn:
		labels:

	Returns:

	"""
	Y = []
	anchors = []
	positives = []
	negatives = []
	for i in range(len(data_real)):
		x_anchor = data_real[i, :, :]

		y = labels_real[i]
		indices_for_pos = np.squeeze(np.where(labels_syn == y))
		indices_for_neg = np.squeeze(np.where(labels_syn != y))
		for j in range(len(indices_for_pos)):
			x_positive = data_syn[indices_for_pos[j], :, :]
			ind_for_neg = indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]
			x_negative = data_syn[ind_for_neg, :, :]
			anchors.append(x_anchor)
			positives.append(x_positive)
			negatives.append(x_negative)
			
			label_n = labels_syn[ind_for_neg]
			y_a = np.zeros(class_num)
			y_a[int(y)] = 1
			y_n = np.zeros(class_num)
			y_n[int(label_n)] = 1
			
			Y.append(np.concatenate([np.zeros(3*emb_size), y_a, y_a, y_n]))
			
	print(f"number of triplets: {len(anchors)}")
	return [np.array(anchors), np.array(positives), np.array(negatives)], np.array(Y)
