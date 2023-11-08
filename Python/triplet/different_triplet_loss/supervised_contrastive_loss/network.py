"""
in training phase 1, training with 2 views' real and synthetic data
"""
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from triplet.utils import LossHistory, CustomizedEarlyStopping
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from utils import DataGeneratorBH_realAnchor, triplet_loss_bh, DataGeneratorBH_realAnchor_randomBatch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import load_model
from triplet.different_triplet_loss.config import *
from tensorflow.keras import regularizers


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

embedding_model = tf.keras.models.Sequential([
	Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer=HeNormal(), bias_initializer=HeNormal(), kernel_regularizer=regularizers.l2()),
	LeakyReLU(),
	MaxPooling2D((2, 2), padding='same'),
	Conv2D(64, (5, 5), padding='same', strides=(2, 2), kernel_initializer=HeNormal(), bias_initializer=HeNormal(), kernel_regularizer=regularizers.l2()),
	LeakyReLU(),
	MaxPooling2D((2, 2), padding='same'),
	Flatten(),
	# Dropout(0.2),
	Dense(256, kernel_initializer=HeNormal(), bias_initializer=HeNormal(), kernel_regularizer=regularizers.l2()),
	LeakyReLU(),
	# Dropout(0.2),
	Dense(256, kernel_initializer=HeNormal(), bias_initializer=HeNormal(), kernel_regularizer=regularizers.l2()),
	LeakyReLU(),
	# Dropout(0.2),
	Dense(emb_size, kernel_initializer=HeNormal(), bias_initializer=HeNormal())
])

# two-way contrastive learning
input_shape = (28, 52, 1)
input_anchor = tf.keras.layers.Input(shape=input_shape)
input_positive = tf.keras.layers.Input(shape=input_shape)

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive], output)


optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss=triplet_loss_bh, optimizer=optimizer)
# net = load_model(model_path_old + "model_weights_16.hdf5", custom_objects={'triplet_loss_bh': triplet_loss_bh})

embed_model = Model(
  inputs=net.layers[2].input,
  outputs=net.layers[2].output
)

net.summary()

# model_path = "../../../models/triplet_v46/"
data_path = "../../../../models/triplet/"
if not os.path.exists(data_path):
	print("{} does not exist.".format(data_path))
	exit(0)
# os.mkdir(model_path)

# net = load_model(model_path + "model_weights_16.hdf5", custom_objects={'triplet_loss_bh': triplet_loss_bh})

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

# path_v3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
# X_real = np.vstack((np.load(os.path.join(path_v3, "X_4.npy")), X_real))
# y_real = np.concatenate((np.load(os.path.join(path_v3, "Y_4.npy")) - 1, y_real))
# X_syn = np.vstack((np.load(os.path.join(path_v3, "X_4_syn.npy")), X_syn))
# y_syn = np.concatenate((np.load(os.path.join(path_v3, "Y_4_syn.npy")) - 1, y_syn))

# normalization
X_real = (X_real - np.mean(X_real)) / (np.std(X_real))
X_syn = (X_syn - np.mean(X_syn)) / (np.std(X_syn))

real_data_size = 0.8

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
# X_train_r, _, y_train_r, _ = train_test_split(X_train_r, y_train_r, train_size=real_data_size, random_state=42)
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

train_generator = DataGeneratorBH_realAnchor(embed_model, P=P, K=K, class_num=class_num, data_real=X_train_r, labels_real=y_train_r, data_syn=X_train_s, labels_syn=y_train_s)
test_generator = DataGeneratorBH_realAnchor(embed_model, P=P, K=K, class_num=class_num, data_real=X_test_r, labels_real=y_test_r, data_syn=X_test_s, labels_syn=y_test_s)
val_generator = DataGeneratorBH_realAnchor(embed_model, P=P, K=K, class_num=class_num, data_real=X_val_r, labels_real=y_val_r, data_syn=X_val_s, labels_syn=y_val_s)
# train_generator = DataGeneratorBH_realAnchor_randomBatch(batch_size=batch_size, class_num=class_num, data_real=X_train_r, labels_real=y_train_r, data_syn=X_train_s, labels_syn=y_train_s)
# test_generator = DataGeneratorBH_realAnchor_randomBatch(batch_size=batch_size, class_num=class_num, data_real=X_test_r, labels_real=y_test_r, data_syn=X_test_s, labels_syn=y_test_s)
# val_generator = DataGeneratorBH_realAnchor_randomBatch(batch_size=batch_size, class_num=class_num, data_real=X_val_r, labels_real=y_val_r, data_syn=X_val_s, labels_syn=y_val_s)

# callbacks
history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
                      os.path.join(model_path, f"loss_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss',
                                   save_best_only=True)

history = net.fit(train_generator,
                  epochs=epochs,
                  validation_data=val_generator,
                  callbacks=[history, early_stopping, model_checkpoint])

# net.save(os.path.join(model_path, 'model_weights_16.hdf5'))
print(1)

