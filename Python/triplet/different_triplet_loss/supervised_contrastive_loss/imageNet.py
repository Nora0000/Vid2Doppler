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
from utils import DataGeneratorBH_realAnchor, triplet_loss_bh
from config import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, Input, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import load_model
from config import *


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the pre-trained layers
base_model.trainable = False

# Create a projection head
projection_head = tf.keras.Sequential([
    Input(shape=(7, 7, 2048)),
    Dense(256, activation='relu'),
    BatchNormalization(),
	# Dense(256, activation='relu'),
    # BatchNormalization(),
    Dense(128, activation='relu')
])

# two-way contrastive learning
input_shape = (224, 224, 3)
input_anchor = tf.keras.layers.Input(shape=input_shape)
input_positive = tf.keras.layers.Input(shape=input_shape)

x1 = base_model(input_anchor)
x2 = base_model(input_positive)

embedding1 = projection_head(x1)
embedding2 = projection_head(x2)

embedding_anchor = tf.math.l2_normalize(embedding1)
embedding_positive = tf.math.l2_normalize(embedding2)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive], output)


optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss=triplet_loss_bh, optimizer=optimizer)

embed_model = Model(
  inputs=net.layers[2].input,
  outputs=net.layers[2].output
)

net.summary()

# model_path = "../../../models/triplet_v46/"
data_path = "../../../models/triplet/"
if not os.path.exists(data_path):
	print("{} does not exist.".format(data_path))
	exit(0)
# os.mkdir(model_path)

# net = load_model(model_path + "model_weights_32.hdf5", custom_objects={'triplet_loss_bh': triplet_loss_bh})

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

# normalization
X_real = (X_real - np.mean(X_real)) / (np.std(X_real))
X_syn = (X_syn - np.mean(X_syn)) / (np.std(X_syn))

X_real = np.stack((X_real, )*3, axis=-1)
X_syn = np.stack((X_syn,)*3, axis=-1)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
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

# callbacks
history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
                      os.path.join(model_path, f"loss_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_loss', patience=20, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss',
                                   save_best_only=True)

history = net.fit(train_generator,
                  epochs=epochs,
                  validation_data=val_generator,
                  callbacks=[history, early_stopping, model_checkpoint])

print(1)
# net.save(os.path.join(model_path, 'model_weights.hdf5'))


# while batch_size >= 2:
# 	batch_size = batch_size // 2
# 	print("batch size: {}".format(batch_size))
# 	history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"),
# 	                      os.path.join(model_path, f"loss_{batch_size}.npy"))
# 	model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss',
# 	                                   save_best_only=True)
# 	history = net.fit(X_train, np.zeros((len(X_train[0]), emb_size * 3)), steps_per_epoch=len(X_train[0]) // batch_size,
# 	                  epochs=epochs,
# 	                  validation_data=(X_val, np.zeros((len(X_val[0]), 3 * emb_size))),
# 	                  callbacks=[history, early_stopping, model_checkpoint])



