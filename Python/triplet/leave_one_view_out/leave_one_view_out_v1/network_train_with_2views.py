"""
in training phase 1, training with 2 views' real and synthetic data
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
from triplet.utils import triplet_loss, LossHistory, CustomizedEarlyStopping
import os
from sklearn.model_selection import train_test_split
from triplet.dataset import tripletDataset_balanced_realAnchor, tripletDataset_allTriplets, tripletDataset_balanced_synAnchor, \
	tripletDataset_all_realAnchor
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from prepare_data import augment_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

emb_size = 128

embedding_model = tf.keras.models.Sequential([
	Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2)),
	MaxPooling2D((2, 2), padding='same'),
	Conv2D(64, (5, 5), activation='relu', padding='same', strides=(2, 2)),
	MaxPooling2D((2, 2), padding='same'),
	Flatten(),
	Dropout(0.2),
	Dense(256, activation="relu"),
	Dropout(0.2),
	Dense(256, activation="relu"),
	Dropout(0.2),
	Dense(emb_size),
])

input_shape = (28, 52, 1)
input_anchor = tf.keras.layers.Input(shape=input_shape)
input_positive = tf.keras.layers.Input(shape=input_shape)
input_negative = tf.keras.layers.Input(shape=input_shape)

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)

alpha = 0.2

batch_size = 32
epochs = 100

lr_schedule = ExponentialDecay(
	initial_learning_rate=5e-4,
	decay_steps=10000,
	decay_rate=0.9)
optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss=triplet_loss, optimizer=optimizer)

net.summary()

model_path = "../../../models/triplet_cross_view_v46_5_real0.8/"
data_path = "../../../models/triplet/"
if not os.path.exists(data_path):
	print("{} does not exist.".format(data_path))
	exit(0)
# os.mkdir(model_path)

if not os.path.exists(model_path):
	os.mkdir(model_path)

path_v1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
X_real = np.load(os.path.join(path_v1, "X_4.npy"))
y_real = np.load(os.path.join(path_v1, "Y_4.npy")) - 1
X_syn = np.load(os.path.join(path_v1, "X_4_syn.npy"))
y_syn = np.load(os.path.join(path_v1, "Y_4_syn.npy")) - 1

path_v2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
X_real = np.vstack((np.load(os.path.join(path_v2, "X_4.npy")), X_real))
y_real = np.concatenate((np.load(os.path.join(path_v2, "Y_4.npy")) - 1, y_real))
X_syn = np.vstack((np.load(os.path.join(path_v2, "X_4_syn.npy")), X_syn))
y_syn = np.concatenate((np.load(os.path.join(path_v2, "Y_4_syn.npy")) - 1, y_syn))

# path_v3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
# X_syn = np.vstack((np.load(os.path.join(path_v3, "X_4_syn.npy")), X_syn))
# y_syn = np.concatenate((np.load(os.path.join(path_v3, "Y_4_syn.npy")) - 1, y_syn))


X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
# # select one sample from each class randomly
# selected = []
# for i in range(5):
# 	selected_t = np.random.choice(np.where(y_train_r == i)[0], 1)
# 	selected.extend(selected_t)
# X_train_r = X_train_r[selected, :, :]
# y_train_r = np.arange(0, 5)

# rem = [i not in selected for i in range(len(X_train_r))]
# X_train_r, _, y_train_r, _ = train_test_split(X_train_r, y_train_r, train_size=0.4, random_state=42)
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_r, y_train_r, test_size=0.1, random_state=42)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1, random_state=42)

np.save(os.path.join(model_path, "X_train_real.npy"), X_train_r)
np.save(os.path.join(model_path, "X_train_syn.npy"), X_train_s)
np.save(os.path.join(model_path, "X_test_real.npy"), X_test_r)
np.save(os.path.join(model_path, "X_test_syn.npy"), X_test_s)
np.save(os.path.join(model_path, "Y_train_real.npy"), y_train_r)
np.save(os.path.join(model_path, "Y_train_syn.npy"), y_train_s)
np.save(os.path.join(model_path, "Y_test_real.npy"), y_test_r)
np.save(os.path.join(model_path, "Y_test_syn.npy"), y_test_s)
# X_real_new = np.load(os.path.join(path_v3, "X_4.npy"))
# y_real_new = np.load(os.path.join(path_v3, "Y_4.npy")) - 1
# _, X_real_new_test, _, Y_real_new_test = train_test_split(X_real_new, y_real_new, test_size=0.2, random_state=42)
# np.save(os.path.join(model_path, "X_test_real_leaveout.npy"), X_real_new_test)
# np.save(os.path.join(model_path, "Y_test_real_leaveout.npy"), Y_real_new_test)


# X_train = tripletDataset_v1(X_train_r, X_train_s, y_train_r)
# X_test = tripletDataset_v1(X_test_r, X_test_s, y_test_r)
# X_val = tripletDataset_v1(X_val_r, X_val_s, y_val_r)
X_train = tripletDataset_balanced_realAnchor(X_train_r, y_train_r, X_train_s, y_train_s)
X_test = tripletDataset_balanced_realAnchor(X_test_r, y_test_r, X_test_s, y_test_s)
X_val = tripletDataset_balanced_realAnchor(X_val_r, y_val_r, X_val_s, y_val_s)

print("number of triplets. train: {}, test: {}, val: {}".format(len(X_train[0]), len(X_test[0]), len(X_val[0])))
print("data set size. train: {}, test: {}, val: {} GB".format(
	3 * sys.getsizeof(X_train[0]) / 1e9, 3 * sys.getsizeof(X_test[0]) / 1e9, 3 * sys.getsizeof(X_val[0]) / 1e9))

# callbacks
history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
					  os.path.join(model_path, f"loss_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss',
								   save_best_only=True)

history = net.fit(X_train, np.zeros((len(X_train[0]), emb_size * 3)), steps_per_epoch=len(X_train[0]) // batch_size,
				  epochs=epochs,
				  validation_data=(X_val, np.zeros((len(X_val[0]), 3 * emb_size))),
				  callbacks=[history, early_stopping, model_checkpoint])

# net.save(os.path.join(model_path, 'model_weights.hdf5'))


while batch_size >= 2:
	batch_size = batch_size // 2
	print("batch size: {}".format(batch_size))
	history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
						  os.path.join(model_path, f"loss_{batch_size}.npy"))
	model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss',
									   save_best_only=True)
	history = net.fit(X_train, np.zeros((len(X_train[0]), emb_size * 3)), steps_per_epoch=len(X_train[0]) // batch_size,
					  epochs=epochs,
					  validation_data=(X_val, np.zeros((len(X_val[0]), 3 * emb_size))),
					  callbacks=[history, early_stopping, model_checkpoint])



