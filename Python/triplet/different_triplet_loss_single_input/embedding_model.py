import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from triplet.utils import triplet_loss, LossHistory, CustomizedEarlyStopping
import os
from sklearn.model_selection import train_test_split
from triplet.dataset import tripletDataset_allTriplets, DataGenerator
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU
# from prepare_data import augment_data
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class EmbeddingModel(tf.keras.models.Model):
	def __init__(self, emb_size=128, input_shape=(28,52,1)):
		super().__init__()
		
		self.input_layer = tf.keras.layers.Input(shape=input_shape)
		self.cnn1 = Conv2D(32, (5, 5), activation=LeakyReLU(alpha=0.05), padding='same', strides=(2, 2))
		self.pooling = MaxPooling2D((2, 2), padding='same')
		self.cnn2 = Conv2D(64, (5, 5), activation=LeakyReLU(alpha=0.05), padding='same', strides=(2, 2))
		self.flatten = Flatten()
		self.dense1 = Dense(256, activation=LeakyReLU(alpha=0.05))
		self.dense2 = Dense(256, activation=LeakyReLU(alpha=0.05))
		self.dense3 = Dense(emb_size, activation="sigmoid")
		self.dropout = Dropout(0.2)
	
	def call(self, inputs):
		x = self.cnn1(inputs)
		x = self.pooling(x)
		x = self.cnn2(x)
		x = self.pooling(x)
		x = self.flatten(x)
		x = self.dropout(x)
		x = self.dense1(x)
		x = self.dropout(x)
		x = self.dense2(x)
		x = self.dropout(x)
		return self.dense3(x)
	
	
	
	
	