import torch
import os
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import argparse
from helper import root_mean_squared_error, get_spectograms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import h5py
from torch import nn
import seaborn as sns
# import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, schedules
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from prepare_data import augment_data

# Load the Fashion MNIST dataset
# from keras.datasets import fashion_mnist

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

new_model_path = "../../models/cnn_train_4_6syn_test_4_6/"
if not os.path.exists(new_model_path):
	os.mkdir(new_model_path)


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


# Load the data and split it into train and test sets
path4 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x4 = np.load(os.path.join(path4, "X_4.npy"))
y4 = np.load(os.path.join(path4, "Y_4.npy"))
x4_aug, y4_aug = augment_data(x4, y4)
X_train, X_test_4, y_train, y_test_4 = train_test_split(x4_aug, y4_aug, test_size=0.2, random_state=42)

# sensor 6 synthetic data for training
path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x6 = np.load(os.path.join(path6, "X_4_syn.npy"))
y6 = np.load(os.path.join(path6, "Y_4_syn.npy"))
x6_aug, y6_aug = augment_data(x6, y6)

X_train = np.vstack((X_train, x6_aug))
y_train = np.concatenate((y_train, y6_aug)) - 1

# sensor 6 real data for test
path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x6 = np.load(os.path.join(path6, "X_4.npy"))
y6 = np.load(os.path.join(path6, "Y_4.npy"))
x6_aug, y6_aug = augment_data(x6, y6)
_, X_test_6, _, y_test_6 = train_test_split(x6_aug, y6_aug, test_size=0.2, random_state=42)

X_test = np.vstack((X_test_4, X_test_6))
y_test = np.concatenate((y_test_4, y_test_6)) - 1

# Normalize the pixel values to range [0, 1]
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

# Convert class labels to one-hot encoded vectors
num_classes = 5
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Reshape the data to fit the CNN input shape
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

np.save(os.path.join(new_model_path, "X_test.npy"), X_test)
np.save(os.path.join(new_model_path, "y_test.npy"), y_test)

# Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(32, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
# model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
lr_schedule = schedules.ExponentialDecay(
	initial_learning_rate=1e-3,
	decay_steps=100,
	decay_rate=0.999)
opt = Adam(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
best_val_loss = 1e5
history_all = {}
batch_size = 16
for i in range(50):
	print("\n\n---------------------------epoch: {}-------------------------------\n\n".format(i))
	if i > 15:
		batch_size = 8
	if i > 30:
		batch_size = 4
	
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=20, validation_split=0.1,
	                    shuffle=True, workers=1)
	if history_all == {}:
		for key in history.history.keys():
			history_all[key] = history.history[key]
	else:
		mean_val_loss = np.mean(history.history['val_loss'])
		mean_loss = np.mean(history.history['loss'])
		if mean_val_loss <= best_val_loss:
			best_val_loss = mean_val_loss
			model.save(os.path.join(new_model_path, 'cnn.hdf5'))
			with open(os.path.join(new_model_path, 'best_model.txt'), 'a') as f:
				f.write(
					"iteration: {}, learning rate: {}, maximum val_loss in this iteration: {}, mean val_loss in this iteration: {}\n".format(
						i, opt._decayed_lr(tf.float32),
						np.max(history.history['val_loss']), np.mean(history.history['val_loss'])
					))
		for key in history.history.keys():
			history_all[key] = np.hstack((history_all[key], history.history[key]))
	np.save(os.path.join(new_model_path, 'loss.npy'), history_all['loss'])
	np.save(os.path.join(new_model_path, 'val_loss.npy'), history_all['val_loss'])
	plot_history(history_all, new_model_path + 'loss.png')
	# Evaluate the model on the test set
	score = model.evaluate(X_test, y_test, verbose=0)
	print("Test loss:", score[0])
	print("Test accuracy:", score[1])
	with open(os.path.join(new_model_path, 'test_result.txt'), 'a') as f:
		f.write("test loss: {}, test accuracy: {}\n".format(score[0], score[1]))

