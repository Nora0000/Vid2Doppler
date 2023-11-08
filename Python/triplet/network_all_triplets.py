import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import triplet_loss, LossHistory, CustomizedEarlyStopping
import os
from sklearn.model_selection import train_test_split
from dataset import tripletDataset_allTriplets, DataGenerator
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
# from prepare_data import augment_data
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

# alpha = 100



batch_size = 128
epochs = 5

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.98)
optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss=triplet_loss, optimizer=optimizer)

net.summary()


model_path = "../../models/all_triplet_real0.8/"

if not os.path.exists(model_path):
    os.mkdir(model_path)


triplet_info = {
    "train": {
        "filename": "../../models/triplet_v2/train_triplets.dat",
        "shape": (4768320, 28*3, 52)
    },
    "test": {
        "filename": "../../models/triplet_v2/test_triplets.dat",
        "shape": (111688, 28*3, 52)
    },
    "val": {
        "filename": "../../models/triplet_v2/val_triplets.dat",
        "shape": (7186, 28*3, 52)
        }
}

data_path="../../models/triplet/"
X_real = np.load(os.path.join(data_path, "X.npy"))
y_real = np.load(os.path.join(data_path, "Y.npy")) - 1
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
X_train_r, _, y_train_r, _ = train_test_split(X_train_r, y_train_r, train_size=0.4, random_state=42)
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_r, y_train_r, test_size=0.1, random_state=42)

X_syn = np.load(os.path.join(data_path, "X_syn.npy"))
y_syn = np.load(os.path.join(data_path, "Y_syn.npy")) - 1
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1, random_state=42)

mode = "all"
train_generator = DataGenerator(triplet_info["train"], emb_size, batch_size=batch_size, dim=(28, 52), mode=mode,
                                net=net, data_real=X_train_r, labels_real=y_train_r, data_syn=X_train_s, labels_syn=y_train_s)
test_generator = DataGenerator(triplet_info["test"], emb_size, batch_size=batch_size, dim=(28, 52), mode=mode,
                                net=net, data_real=X_test_r, labels_real=y_test_r, data_syn=X_test_s, labels_syn=y_test_s)
val_generator = DataGenerator(triplet_info["val"], emb_size, batch_size=batch_size, dim=(28, 52), mode=mode,
                                net=net, data_real=X_val_r, labels_real=y_val_r, data_syn=X_val_s, labels_syn=y_val_s)

# callbacks
history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"), os.path.join(model_path, f"loss_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss', save_best_only=True)

history = net.fit(train_generator, steps_per_epoch=triplet_info["train"]["shape"][0] // batch_size, epochs=epochs,
                      validation_data=val_generator,
                      callbacks=[history, early_stopping, model_checkpoint])
# history = net.fit(train_generator, steps_per_epoch=5, epochs=epochs,
#                       validation_data=val_generator,
#                       callbacks=[history, early_stopping, model_checkpoint])


while batch_size >= 2:
    batch_size = batch_size // 2
    print("batch size: {}".format(batch_size))
    history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"), os.path.join(model_path, f"loss_{batch_size}.npy"))
    model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss', save_best_only=True)
    history = net.fit(train_generator, steps_per_epoch=triplet_info["train"]["shape"][0] // batch_size, epochs=epochs,
                      validation_data=val_generator,
                      callbacks=[history, early_stopping, model_checkpoint])


