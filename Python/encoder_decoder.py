import torch
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.layers import Dropout, Conv2DTranspose, Activation
from tensorflow.keras import activations, Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.losses import mae
import tensorflow as tf
import os
import numpy as np
import argparse
from helper import root_mean_squared_error, get_spectograms
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import h5py
from prepare_data import augment_data
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.python.keras.models import Model
from prepare_data import interpolate
from torch.nn import UpsamplingBilinear2d
import math
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.ndimage import gaussian_filter1d

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def data_loader(DISCARD_BINS=[15, 16, 17]):
	doppler_gt_all = np.array([])
	doppler_syn_all = np.array([])
	paths = ['../data/2023_05_04/']
	for path in paths:
		path_list = os.listdir(path)
		for path_i in path_list:
			path_i = os.path.join(os.path.abspath(path), path_i)
			if not os.path.isdir(path_i):
				continue
			
			gt = np.load(os.path.join(path_i, "doppler_gt.npy"))
			syn = np.load(os.path.join(path_i, "output/rgb/synth_doppler.npy"))
			
			frames_common = np.load(os.path.join(path_i, "frames_common.npy"))
			# if os.path.isfile(os.path.join(path_i, 'frames_new.npy')):
			# 	frames = np.load(os.path.join(path_i, 'frames_new.npy'), allow_pickle=True)
			# else:
			# 	frames = np.load(os.path.join(path_i, 'frames.npy'), allow_pickle=True)
			frames = np.load(os.path.join(path_i, 'frames.npy'), allow_pickle=True)
			indices = np.isin(frames, frames_common)
			try:
				syn = syn[indices, :]
			except Exception as e:
				print(path_i)
				raise e
			
			gt[:, DISCARD_BINS] = np.zeros((gt.shape[0], len(DISCARD_BINS)))
			syn[:, DISCARD_BINS] = np.zeros((syn.shape[0], len(DISCARD_BINS)))
			
			gt = get_spectograms(gt, 3, 24, synthetic=True, zero_pad=True)
			gt = gt.astype("float32")
			syn = get_spectograms(syn, 3, 24, synthetic=True, zero_pad=True)
			syn = syn.astype("float32")
			if len(doppler_gt_all) == 0:
				doppler_gt_all = np.copy(gt)
				doppler_syn_all = np.copy(syn)
			else:
				doppler_gt_all = np.vstack((doppler_gt_all, gt))
				doppler_syn_all = np.vstack((doppler_syn_all, syn))
	# # normalization
	# doppler_syn_all = (doppler_syn_all - min_synth_dopVal) / (max_synth_dopVal - min_synth_dopVal)
	# doppler_gt_all = (doppler_gt_all - min_dopVal) / (max_dopVal - min_dopVal)
	return np.array(doppler_syn_all)[:, :, :, np.newaxis], np.array(doppler_gt_all)[:, :, :, np.newaxis]


def create_model(model_path):
	old_model = load_model(model_path + "autoencoder_weights.hdf5",
	                       custom_objects={'root_mean_squared_error': root_mean_squared_error})
	
	model = Sequential()
	model.add(old_model.layers[0])
	for layer in old_model.layers[1].layers[1:]:
		if "dense" in layer.name:
			model.add(Dropout(0.4))
		model.add(layer)
	
	model.summary()
	lr_schedule = schedules.ExponentialDecay(
		initial_learning_rate=0.5e-3,
		decay_steps=10000,
		decay_rate=0.999)
	model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=root_mean_squared_error)
	return model


def lr_schedule(epoch):  # drop by 1/2 every 10 epochs
	initial_lr = 0.5e-3
	drop = 0.5
	epochs_drop = 10
	lr = initial_lr * math.pow(drop, math.floor((1 + epoch / 10) / epochs_drop))
	return lr


def main(args):
	path4 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
	x4 = np.load(os.path.join(path4, "X_4.npy"))
	# y4 = np.load(os.path.join(path4, "Y_4.npy"))
	x4_syn = np.load(os.path.join(path4, "X_4_syn.npy"))
	# y4_syn = np.load(os.path.join(path4, "Y_4_syn.npy"))
	path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
	x6 = np.load(os.path.join(path6, "X_4.npy"))
	# y6 = np.load(os.path.join(path6, "Y_4.npy"))
	x6_syn = np.load(os.path.join(path6, "X_4_syn.npy"))
	# y6_syn = np.load(os.path.join(path6, "Y_4_syn.npy"))
	X = np.vstack((x4_syn, x6_syn))
	y = np.vstack((x4, x6))
	# X = x6_syn
	# y = x6
	X_aug, _ = augment_data(X, [1] * len(X))
	y_aug, _ = augment_data(y, [1] * len(y))
	X_aug = UpsamplingBilinear2d(size=(32, 52))(torch.tensor(X_aug[np.newaxis, :, :, :])).numpy()[0, :, :, :]
	y_aug = UpsamplingBilinear2d(size=(32, 52))(torch.tensor(y_aug[np.newaxis, :, :, :])).numpy()[0, :, :, :]
	X_aug = np.delete(X_aug, [14, 15, 16, 17], axis=1)
	y_aug = np.delete(y_aug, [14, 15, 16, 17], axis=1)
	for i in range(X_aug.shape[0]):
		for j in range(52):
			X_aug[i, :, j] = gaussian_filter1d(X_aug[i, :, j], 3)
	X_aug = (X_aug - np.min(X_aug)) / (np.max(X_aug) - np.min(X_aug))
	y_aug = (y_aug - np.min(y_aug)) / (np.max(y_aug) - np.min(y_aug))
	X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
	# X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
	# X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
	# y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))
	# y_test = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
	# X_val = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val))
	# y_val = (y_val - np.min(y_val)) / (np.max(y_val) - np.min(y_val))
	
	model_path = "../models/encoder_s4_s6_larger_model/"
	if not os.path.exists(model_path):
		os.mkdir(model_path)
	
	np.save(os.path.join(model_path, "X_test.npy"), X_test)
	np.save(os.path.join(model_path, "y_test.npy"), y_test)
	np.save(os.path.join(model_path, "X_train.npy"), X_train)
	np.save(os.path.join(model_path, "y_train.npy"), y_train)
	
	# Define the input shape
	input_shape = (28, 52, 1)
	
	# Encoder
	input_img = Input(shape=input_shape)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(input_img)
	# x = Dropout(0.2)(x)
	# x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	# x = Dropout(0.2)(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	# x = Dropout(0.2)(x)
	# encoded = MaxPooling2D((2, 2), padding='same')(x)
	x = Flatten()(x)
	encoded = Dense(5824, activation="relu")(x)
	# encoded = Dropout(0.2)(encoded)
	
	# Decoder
	x = Dense(5824, activation="relu")(encoded)
	# x = Dropout(0.2)(x)
	x = Reshape(target_shape=(7, 13, 64))(x)
	# x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
	x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
	x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	# x = Dropout(0.2)(x)
	# x = UpSampling2D((2, 2))(x)
	x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(1, 1))(x)
	# x = Dropout(0.2)(x)
	# x = UpSampling2D((2, 2))(x)
	decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1))(x)
	
	# Create the autoencoder model
	autoencoder = Model(input_img, decoded)
	
	# Compile the model
	# lr_schedule = schedules.ExponentialDecay(
	# 	initial_learning_rate=1e-4,
	# 	decay_steps=10000,
	# 	decay_rate=0.999)
	lr = 5e-5
	# opt = Adam(lr)
	# lr_scheduler = LearningRateScheduler(lr_schedule)
	autoencoder.compile(optimizer=Adam(lr), loss=root_mean_squared_error)
	
	# Print the model summary
	autoencoder.summary()
	
	history_all = {}
	best_val_loss = 100
	batch_size = 16
	for i in range(1000):
		
		if 0 < i <= 10:
			lr = 5e-5 * (2 * i)
		else:
			lr = max(lr / (i - 3), 0.5e-4)
			batch_size = max(int(batch_size / 2), 1)
		autoencoder.compile(optimizer=Adam(lr), loss=root_mean_squared_error)
		print("\n\n---------------------------epoch: {}, lr: {} -------------------------------\n\n".format(i, lr))
		
		# def lr_schedule(epoch):  # drop by 1/2 every 10 epochs
		# 	initial_lr = 0.5e-3
		# 	drop = 0.5
		# 	epochs_drop = 10
		# 	lr = initial_lr * math.pow(drop, math.floor((1 + epoch / 10) / epochs_drop))
		# 	return lr
		# lr = lr * math.pow(0.5, math.floor(i / 5))
		# autoencoder.compile(optimizer=Adam(lr), loss=root_mean_squared_error)
		# if i > 5:
		# 	batch_size = 8
		# if i > 10:
		# 	batch_size = 4
		# if i > 15:
		# 	batch_size = 2
		# history = model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(X_test, y_test), shuffle=True, workers=1, callbacks=[checkpoint_cb, tensorboard_cb])
		history = autoencoder.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(X_val, y_val),
		                          shuffle=True, workers=1)
		if history_all == {}:
			for key in history.history.keys():
				history_all[key] = history.history[key]
		else:
			mean_val_loss = np.mean(history.history['val_loss'])
			mean_loss = np.mean(history.history['loss'])
			if mean_val_loss <= best_val_loss:
				best_val_loss = mean_val_loss
				autoencoder.save(os.path.join(model_path, 'autoencoder_weights.hdf5'))
			# with open(os.path.join(model_path, 'best_model.txt'), 'a') as f:
			# 	f.write("iteration: {}, learning rate: {}, maximum val_loss in this iteration: {}, mean val_loss in this iteration: {}\n".format(
			# 		i, opt._decayed_lr(tf.float32),
			# 		np.max(history.history['val_loss']), np.mean(history.history['val_loss'])
			# 	))
			for key in history.history.keys():
				history_all[key] = np.hstack((history_all[key], history.history[key]))
		np.save(os.path.join(model_path, 'loss.npy'), history_all['loss'])
		np.save(os.path.join(model_path, 'val_loss.npy'), history_all['val_loss'])
		plot_history(history_all, model_path + 'loss.png')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--model_path', type=str, help='Path to DL models', default='../models/')
	
	args = parser.parse_args()
	
	main(args)



