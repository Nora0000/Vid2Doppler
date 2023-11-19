"""
train the classifier with synthetic data and finetune the last layer with real data.
For classifier, use the same structure with contrastive learning.
"""
import sys
import torch
from torch.nn import UpsamplingBilinear2d
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from triplet.utils import triplet_loss, LossHistory, CustomizedEarlyStopping, AccuracyHistory
import os
from sklearn.model_selection import train_test_split
from triplet.dataset import tripletDataset_balanced_realAnchor, tripletDataset_allTriplets, tripletDataset_balanced_synAnchor, \
	tripletDataset_all_realAnchor
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from prepare_data import augment_data
from config import *
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# emb_size = 1500
input_shape = (28, 52, 1)


net = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=input_shape),
	Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2)),
	MaxPooling2D((2, 2), padding='same'),
	Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2)),
	MaxPooling2D((2, 2), padding='same'),
	Flatten(),
	Dropout(0.2),
	Dense(6, activation="relu"),
	Dropout(0.2),
	# Dense(128, activation="relu"),
	# Dropout(0.5),
	Dense(class_num),
])

batch_size = 32
epochs = 1000

lr_schedule = ExponentialDecay(
	initial_learning_rate=5e-4,
	decay_steps=10000,
	decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

net.summary()

model_path = "/home/mengjingliu/Vid2Doppler/models/triplet_cross_view_real0.64/with_synthetic_of_unseen_view/finetune_v46_5/"


if not os.path.exists(model_path):
	os.mkdir(model_path)

# path_v1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
# X_real = np.load(os.path.join(path_v1, "X_4.npy"))
# y_real = np.load(os.path.join(path_v1, "Y_4.npy")) - 1
# X_syn = np.load(os.path.join(path_v1, "X_4_syn.npy"))
# y_syn = np.load(os.path.join(path_v1, "Y_4_syn.npy")) - 1
#
# path_v2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
# X_real = np.vstack((np.load(os.path.join(path_v2, "X_4.npy")), X_real))
# y_real = np.concatenate((np.load(os.path.join(path_v2, "Y_4.npy")) - 1, y_real))
# X_syn = np.vstack((np.load(os.path.join(path_v2, "X_4_syn.npy")), X_syn))
# y_syn = np.concatenate((np.load(os.path.join(path_v2, "Y_4_syn.npy")) - 1, y_syn))
#
# path_v3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
# X_syn = np.vstack((np.load(os.path.join(path_v3, "X_4_syn.npy")), X_syn))
# y_syn = np.concatenate((np.load(os.path.join(path_v3, "Y_4_syn.npy")) - 1, y_syn))


# X_real = (X_real - np.mean(X_real)) / np.std(X_real)
# X_syn = (X_syn - np.mean(X_syn)) / np.std(X_syn)

# X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
# X_train_r, _, y_train_r, _ = train_test_split(X_train_r, y_train_r, train_size=0.2, random_state=42)
# X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_r, y_train_r, test_size=0.1, random_state=42)
#
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
# X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1, random_state=42)

path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x1 = np.load(os.path.join(path1, "X_4.npy"))[:,:,:, np.newaxis]
y1 = np.load(os.path.join(path1, "Y_4.npy"))-1
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, random_state=42)
x1_train, _, y1_train, _ = train_test_split(x1_train, y1_train, train_size=data_size, random_state=42)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size=0.9, random_state=42)
# synthetic
x1_syn = np.load(os.path.join(path1, "X_4_syn.npy"))[:,:,:, np.newaxis]
y1_syn = np.load(os.path.join(path1, "Y_4_syn.npy"))-1
x1_syn, _, y1_syn, _ = train_test_split(x1_syn, y1_syn, train_size=0.8, random_state=42)
x1_syn, x1_syn_val, y1_syn, y1_syn_val = train_test_split(x1_syn, y1_syn, train_size=0.9, random_state=42)

path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x2 = np.load(os.path.join(path2, "X_4.npy"))[:,:,:, np.newaxis]
y2 = np.load(os.path.join(path2, "Y_4.npy"))-1
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, random_state=42)
x2_train, _, y2_train, _ = train_test_split(x2_train, y2_train, train_size=data_size, random_state=42)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train, y2_train, train_size=0.9, random_state=42)
# synthetic
x2_syn = np.load(os.path.join(path2, "X_4_syn.npy"))[:,:,:, np.newaxis]
y2_syn = np.load(os.path.join(path2, "Y_4_syn.npy"))-1
x2_syn, _, y2_syn, _ = train_test_split(x2_syn, y2_syn, train_size=0.8, random_state=42)
x2_syn, x2_syn_val, y2_syn, y2_syn_val = train_test_split(x2_syn, y2_syn, train_size=0.9, random_state=42)

path3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
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

# callbacks
loss_history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
					  os.path.join(model_path, f"loss_{batch_size}.npy"))
acc_history = AccuracyHistory(os.path.join(model_path, f"acc_{batch_size}.png"),
					  os.path.join(model_path, f"acc_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_accuracy', patience=20, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_accuracy',
								   save_best_only=True)

net.fit(X_train_s, y_train_s,
	  epochs=epochs,
      # batch_size=batch_size,
	  validation_data=(X_val_s, y_val_s),
	  callbacks=[loss_history, acc_history, early_stopping, model_checkpoint])

# freeze layers
for layer in net.layers[:-1]:
	layer.trainable = False

loss_history = LossHistory(os.path.join(model_path, f"loss_{batch_size}_finetune.png"),
					  os.path.join(model_path, f"loss_{batch_size}_finetune.npy"))
acc_history = AccuracyHistory(os.path.join(model_path, f"acc_{batch_size}_finetune.png"),
					  os.path.join(model_path, f"acc_{batch_size}_finetune.npy"))
net.fit(X_train_r, y_train_r,
	  epochs=epochs,
      # batch_size=batch_size,
	  validation_data=(X_val_r, y_val_r),
	  callbacks=[loss_history, acc_history, early_stopping, model_checkpoint])
