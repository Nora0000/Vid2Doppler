import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import plot_history, triplet_loss, LossHistory, CustomizedEarlyStopping
import os
from sklearn.model_selection import train_test_split
from dataset import tripletDataset, tripletDataset_v2, data_generator
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



batch_size = 1024
epochs = 100

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.98)
optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss=triplet_loss, optimizer=optimizer)

net.summary()


model_path = "../../models/triplet_v2/"
data_path = "../../models/triplet/"
if not os.path.exists(data_path):
    print("{} does not exist.".format(data_path))
    exit(0)
    # os.mkdir(model_path)

if not os.path.exists(model_path):
    os.mkdir(model_path)

X_real = np.load(os.path.join(data_path, "X.npy"))
y_real = np.load(os.path.join(data_path, "Y.npy")) - 1
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_r, y_train_r, test_size=0.1, random_state=42)
# X_train_r_aug, y_train_r_aug = augment_data(X_train_r, y_train_r)
# X_val_r_aug, y_val_r_aug = augment_data(X_val_r, y_val_r)


X_syn = np.load(os.path.join(data_path, "X_syn.npy"))
y_syn = np.load(os.path.join(data_path, "Y_syn.npy")) - 1
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1, random_state=42)
# X_train_s_aug, y_train_s_aug = augment_data(X_train_s, y_train_s)
# X_val_s_aug, y_val_s_aug = augment_data(X_val_s, y_val_s)

# X_train = tripletDataset(X_train_r, X_train_s, y_train_r)
# X_test = tripletDataset(X_test_r, X_test_s, y_test_r)
# X_val = tripletDataset(X_val_r, X_val_s, y_val_r)
X_train = tripletDataset_v2(X_train_r, X_train_s, y_train_r)
X_test = tripletDataset_v2(X_test_r, X_test_s, y_test_r)
X_val = tripletDataset_v2(X_val_r, X_val_s, y_val_r)

print("number of triplets. train: {}, test: {}, val: {}".format(len(X_train[0]), len(X_test[0]), len(X_val[0])))
print("data set size. train: {}, test: {}, val: {} GB".format(
    3 * sys.getsizeof(X_train[0])/1e9, 3 * sys.getsizeof(X_test[0])/1e9, 3 * sys.getsizeof(X_val[0])/1e9))

train_generator = data_generator(X_train, batch_size, emb_size)

# callbacks
history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"), os.path.join(model_path, f"loss_{batch_size}.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss', save_best_only=True)

history = net.fit(train_generator, steps_per_epoch=len(X_train[0]) // batch_size, epochs=epochs,
                      validation_data=(X_val, np.zeros((len(X_val[0]), 3*emb_size))),
                      callbacks=[history, early_stopping, model_checkpoint])


while batch_size >= 1:
    batch_size = batch_size // 2
    print("batch size: {}".format(batch_size))
    history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"), os.path.join(model_path, f"loss_{batch_size}.npy"))
    model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss', save_best_only=True)
    history = net.fit(train_generator, steps_per_epoch=len(X_train[0]) // batch_size, epochs=epochs,
                          validation_data=(X_val, np.zeros((len(X_val[0]), 3*emb_size))),
                          callbacks=[history, early_stopping, model_checkpoint])



#
# best_val_loss = 100
# history_all = {}



# for i in range(100):
#     print("\n\n---------------------------epoch: {} -------------------------------\n\n".format(i))
#     history = net.fit(train_generator, steps_per_epoch=len(X_train[0]) // batch_size, epochs=epochs,
#                       validation_data=(X_val, np.zeros((len(X_val[0]), 3*emb_size))),
#                       callbacks=[history, ])

    # if history_all == {}:
    #     for key in history.history.keys():
    #         history_all[key] = history.history[key]
    # else:
    #     mean_val_loss = np.mean(history.history['val_loss'])
    #     mean_loss = np.mean(history.history['loss'])
    #     if mean_val_loss <= best_val_loss:
    #         best_val_loss = mean_val_loss
    #         net.save(os.path.join(model_path, 'model_weights.hdf5'))
    #     # with open(os.path.join(model_path, 'best_model.txt'), 'a') as f:
    #     # 	f.write("iteration: {}, learning rate: {}, maximum val_loss in this iteration: {}, mean val_loss in this iteration: {}\n".format(
    #     # 		i, opt._decayed_lr(tf.float32),
    #     # 		np.max(history.history['val_loss']), np.mean(history.history['val_loss'])
    #     # 	))
    #     for key in history.history.keys():
    #         history_all[key] = np.hstack((history_all[key], history.history[key]))
    # np.save(os.path.join(model_path, 'loss.npy'), history_all['loss'])
    # np.save(os.path.join(model_path, 'val_loss.npy'), history_all['val_loss'])
    # plot_history(history_all, model_path + 'loss.png')
