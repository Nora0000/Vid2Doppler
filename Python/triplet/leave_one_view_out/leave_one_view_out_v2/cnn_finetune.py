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
from triplet.utils import LossHistory, CustomizedEarlyStopping, AccuracyHistory
from sklearn.metrics import accuracy_score

# Load the Fashion MNIST dataset
# from keras.datasets import fashion_mnist

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = "/home/mengjingliu/Vid2Doppler/models/triplet_cross_view_real0.8/with_synthetic_of_unseen_view/finetune_v45_6/"
if not os.path.exists(model_path):
	os.mkdir(model_path)

data_size = 0.2
num_classes = 5

path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x1 = np.load(os.path.join(path1, "X_4.npy"))[:,:,:, np.newaxis]
y1 = np.load(os.path.join(path1, "Y_4.npy"))-1
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, random_state=42)
# x1_train, _, y1_train, _ = train_test_split(x1_train, y1_train, train_size=data_size, random_state=42)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size=0.9, random_state=42)
# synthetic
x1_syn = np.load(os.path.join(path1, "X_4_syn.npy"))[:,:,:, np.newaxis]
y1_syn = np.load(os.path.join(path1, "Y_4_syn.npy"))-1
x1_syn, _, y1_syn, _ = train_test_split(x1_syn, y1_syn, train_size=0.8, random_state=42)
x1_syn, x1_syn_val, y1_syn, y1_syn_val = train_test_split(x1_syn, y1_syn, train_size=0.9, random_state=42)

path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
x2 = np.load(os.path.join(path2, "X_4.npy"))[:,:,:, np.newaxis]
y2 = np.load(os.path.join(path2, "Y_4.npy"))-1
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, random_state=42)
# x2_train, _, y2_train, _ = train_test_split(x2_train, y2_train, train_size=data_size, random_state=42)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train, y2_train, train_size=0.9, random_state=42)
# synthetic
x2_syn = np.load(os.path.join(path2, "X_4_syn.npy"))[:,:,:, np.newaxis]
y2_syn = np.load(os.path.join(path2, "Y_4_syn.npy"))-1
x2_syn, _, y2_syn, _ = train_test_split(x2_syn, y2_syn, train_size=0.8, random_state=42)
x2_syn, x2_syn_val, y2_syn, y2_syn_val = train_test_split(x2_syn, y2_syn, train_size=0.9, random_state=42)

path3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x3 = np.load(os.path.join(path3, "X_4.npy"))[:,:,:, np.newaxis]
y3 = np.load(os.path.join(path3, "Y_4.npy"))-1
_, x3_test, _, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)
# synthetic
x3_syn = np.load(os.path.join(path3, "X_4_syn.npy"))[:,:,:, np.newaxis]
y3_syn = np.load(os.path.join(path3, "Y_4_syn.npy"))-1
x3_syn, _, y3_syn, _ = train_test_split(x3_syn, y3_syn, train_size=0.8, random_state=42)
x3_syn, x3_syn_val, y3_syn, y3_syn_val = train_test_split(x3_syn, y3_syn, train_size=0.9, random_state=42)

X_train_r = np.vstack([x1_train, x2_train])
X_train_s = np.vstack([x1_syn, x2_syn, x3_syn])
y_train_r = to_categorical(np.concatenate([y1_train, y2_train]))
y_train_s = to_categorical(np.concatenate([y1_syn, y2_syn, y3_syn]))

X_test_r = np.vstack([x1_test, x2_test])
y_test_r = to_categorical(np.concatenate([y1_test, y2_test]))

X_val_r = np.vstack([x1_val, x2_val])
X_val_s = np.vstack([x1_syn_val, x2_syn_val, x3_syn_val])
y_val_r = to_categorical(np.concatenate([y1_val, y2_val]))
y_val_s = to_categorical(np.concatenate([y1_syn_val, y2_syn_val, y3_syn_val]))

np.save(os.path.join(model_path, "X_train_real.npy"), X_train_r)
np.save(os.path.join(model_path, "X_train_syn.npy"), X_train_s)
np.save(os.path.join(model_path, "X_test_real.npy"), X_test_r)
np.save(os.path.join(model_path, "Y_train_real.npy"), y_train_r)
np.save(os.path.join(model_path, "Y_train_syn.npy"), y_train_s)
np.save(os.path.join(model_path, "Y_test_real.npy"), y_test_r)
np.save(os.path.join(model_path, "X_test_real_leaveout.npy"), x3_test)
np.save(os.path.join(model_path, "Y_test_real_leaveout.npy"), y3_test)

# Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 52, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
lr_schedule = schedules.ExponentialDecay(
	initial_learning_rate=1e-3,
	decay_steps=100,
	decay_rate=0.999)
opt = Adam(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

batch_size = 32

# Print the model summary
model.summary()
# callbacks
loss_history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
					  os.path.join(model_path, f"loss_{batch_size}.npy"))
acc_history = AccuracyHistory(os.path.join(model_path, f"acc_{batch_size}.png"),
					  os.path.join(model_path, f"acc_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_accuracy', patience=30, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_accuracy',
								   save_best_only=True)

	
model.fit(X_train_s, y_train_s, batch_size=batch_size, epochs=1000,
			shuffle=True, workers=1,
			validation_data=(X_val_s, y_val_s),
          callbacks=[loss_history, acc_history, early_stopping, model_checkpoint])


def test(X_test_r, y_test_r, x3_test, y3_test, model):
	y_test_r = np.argmax(y_test_r, axis=1)
	pre = np.argmax(model.predict(X_test_r), axis=1)
	y3_pre = np.argmax(model.predict(x3_test), axis=1)
	pre_overall = np.concatenate([pre, y3_pre])
	y_overall = np.concatenate([y_test_r, y3_test])
	print(f"accuracy overall: {accuracy_score(y_overall, pre_overall)}")
	print(f"accuracy of two views: {accuracy_score(y_test_r, pre)}")
	print(f"accuracy of leave out view: {accuracy_score(y3_test, y3_pre)}")

# test before finetune
print("------------------before finetune------------------")
test(X_test_r, y_test_r, x3_test, y3_test, model)

# finetune
for layer in model.layers[:-1]:
	layer.trainable = False

loss_history = LossHistory(os.path.join(model_path, f"loss_{batch_size}_finetune.png"),
					  os.path.join(model_path, f"loss_{batch_size}_finetune.npy"))
acc_history = AccuracyHistory(os.path.join(model_path, f"acc_{batch_size}_finetune.png"),
					  os.path.join(model_path, f"acc_{batch_size}_finetune.npy"))
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}_finetune.hdf5'), monitor='val_accuracy',
								   save_best_only=True)
model.fit(X_train_r, y_train_r,
	  epochs=1000,
      batch_size=batch_size,
	  validation_data=(X_val_r, y_val_r),
	  callbacks=[loss_history, acc_history, early_stopping, model_checkpoint])

# test after finetune
print("------------------after finetune------------------")
test(X_test_r, y_test_r, x3_test, y3_test, model)
