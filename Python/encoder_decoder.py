from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.layers import Dropout, Conv2DTranspose, Activation
from tensorflow.keras import activations, Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.losses import mae
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import argparse
from helper import root_mean_squared_error, get_spectograms
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def plot_history(history, filename):
    plt.figure()
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
	return np.array(doppler_syn_all)[:,:,:, np.newaxis], np.array(doppler_gt_all)[:,:,:,np.newaxis]


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


def main(args):
	
	
	
	X, y = data_loader()

	# normalization
	max_x = np.max(X)
	min_x = np.min(X)
	max_y = np.max(y)
	min_y = np.min(y)
	X = (X - min_x) / (max_x - min_x)
	y = (y - min_y) / (max_y - min_y)
	print(max_x)
	print(min_x)
	print(max_y)
	print(min_y)
	#
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
	# np.save('../data/X_test.npy', np.array(X_test))
	# np.save('../data/y_test.npy', np.array(y_test))
	# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
	#
	model_path = args.model_path
	# np.save("../models/scale_vals_new.npy", np.array([max_x, min_x, max_y, min_y]))

	new_model_path = model_path + 'new_encoder_5/'
	if not os.path.exists(new_model_path):
		os.mkdir(new_model_path)
	autoencoder = load_model(model_path + "new_encoder_4/autoencoder_weights.hdf5",
	                         custom_objects={'root_mean_squared_error': root_mean_squared_error})
	# model = Sequential()
	# model.add(autoencoder.layers[0])
	# for layer in autoencoder.layers[1].layers:
	# 	if "input" in layer.name:
	# 		layer._name = 'input_2'
	# 	model.add(layer)
		# if "conv2d_transpose" in layer.name:
		# 	model.add(Dropout(0.2))
		# if "dense" in layer.name:
		# 	model.add(Dropout(0.2))

	# for layer in autoencoder.layers[1:]:
	# 	model.add(layer)
	model = autoencoder
	model.summary()

	# model = load_model('../models/new_encoder/autoencoder_weights.hdf5',
	#                          custom_objects={'root_mean_squared_error': root_mean_squared_error})
	lr_schedule = schedules.ExponentialDecay(
		initial_learning_rate=0.5e-3,
		decay_steps=10000,
		decay_rate=0.999)
	opt = Adam(learning_rate=0.5e-3)
	model.compile(optimizer=opt, loss=root_mean_squared_error)

	# model = create_model(model_path)
	# K.set_value(autoencoder.optimizer.lr, 1e-4)

	# tensorboard_cb = TensorBoard(log_dir=new_model_path + 'logs/', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
	#                             write_images=False, embeddings_freq=0, embeddings_layer_names=None,
	#                             embeddings_metadata=None, embeddings_data=None)

	# checkpoint_cb = ModelCheckpoint(new_model_path, monitor='val_loss', verbose=0, save_best_only=True,
	#                                 save_weights_only=False, mode='max', period=1)

	history_all = {}
	best_val_loss = 100
	batch_size = 4
	for i in range(1000):
		if i > 10:
			batch_size = 2
		if i > 20:
			batch_size = 1
		# if i > 75:
		# 	batch_size = 8
		# history = model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(X_test, y_test), shuffle=True, workers=1, callbacks=[checkpoint_cb, tensorboard_cb])
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_data=(X_test, y_test), shuffle=True, workers=1)
		if history_all == {}:
			for key in history.history.keys():
				history_all[key] = history.history[key]
		else:
			mean_val_loss = np.mean(history.history['val_loss'])
			mean_loss = np.mean(history.history['loss'])
			if mean_val_loss <= best_val_loss:
				best_val_loss = mean_val_loss
				model.save(os.path.join(new_model_path, 'autoencoder_weights.hdf5'))
				with open(os.path.join(new_model_path, 'best_model.txt'), 'a') as f:
					f.write("iteration: {}, learning rate: {}, maximum val_loss in this iteration: {}, mean val_loss in this iteration: {}\n".format(
						i, opt._decayed_lr(tf.float32),
						np.max(history.history['val_loss']), np.mean(history.history['val_loss'])
					))
			for key in history.history.keys():
				history_all[key] = np.hstack((history_all[key], history.history[key]))
		np.save(os.path.join(new_model_path, 'loss.npy'), history_all['loss'])
		np.save(os.path.join(new_model_path, 'val_loss.npy'), history_all['val_loss'])
		plot_history(history_all, new_model_path + 'loss.pdf')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_path', type=str, help='Path to DL models', default='../models/')

	args = parser.parse_args()

	main(args)



