import sys
from config import *
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from triplet.utils import LossHistory, CustomizedEarlyStopping, GradientExplosionCallback
import os
from sklearn.model_selection import train_test_split
from triplet.different_triplet_loss_single_input.embedding_model import EmbeddingModel
from sklearn import svm
from joblib import dump, load
from utils import triple_loss_bh, DataGeneratorBH_realAnchor, DataGeneratorBH_realAnchor_randomBatch
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import load_model, save_model


tf.compat.v1.disable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/'

net = EmbeddingModel()
# net.build(input_shape=(None, 28, 52, 1))
# net.load_weights(model_path + "model_weights.hdf5")
optimizer = Adam(learning_rate=lr_schedule)
net.compile(loss=triple_loss_bh, optimizer=optimizer)



# net.summary()

if not os.path.exists(model_path):
    os.mkdir(model_path)


data_path="../../../models/triplet/"
X_real = np.load(os.path.join(data_path, "X.npy"))
y_real = np.load(os.path.join(data_path, "Y.npy")) - 1
X_real = (X_real - np.mean(X_real)) / (np.std(X_real))

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
np.save(os.path.join(model_path, "X_train_real.npy"), X_train_r)
np.save(os.path.join(model_path, "y_train_real.npy"), y_train_r)
np.save(os.path.join(model_path, "X_test_real.npy"), X_test_r)
np.save(os.path.join(model_path, "y_test_real.npy"), y_test_r)
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_r, y_train_r, test_size=0.1, random_state=42)

X_syn = np.load(os.path.join(data_path, "X_syn.npy"))
y_syn = np.load(os.path.join(data_path, "Y_syn.npy")) - 1
X_syn = (X_syn - np.mean(X_syn)) / (np.std(X_syn))
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
np.save(os.path.join(model_path, "X_train_syn.npy"), X_train_s)
np.save(os.path.join(model_path, "Y_train_syn.npy"), y_train_s)
np.save(os.path.join(model_path, "X_test_syn.npy"), X_test_s)
np.save(os.path.join(model_path, "Y_test_syn.npy"), y_test_s)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_s, y_train_s, test_size=0.1, random_state=42)

# generate batch with P classes and K samples
train_generator = DataGeneratorBH_realAnchor(P=P, K=K, class_num=class_num, data_real=X_train_r, labels_real=y_train_r, data_syn=X_train_s, labels_syn=y_train_s)
test_generator = DataGeneratorBH_realAnchor(P=P, K=K, class_num=class_num, data_real=X_test_r, labels_real=y_test_r, data_syn=X_test_s, labels_syn=y_test_s)
val_generator = DataGeneratorBH_realAnchor(P=P, K=K, class_num=class_num, data_real=X_val_r, labels_real=y_val_r, data_syn=X_val_s, labels_syn=y_val_s)
# generate random batch
# train_generator = DataGeneratorBH_realAnchor_randomBatch(batch_size=batch_size, class_num=class_num, data_real=X_train_r, labels_real=y_train_r, data_syn=X_train_s, labels_syn=y_train_s)
# test_generator = DataGeneratorBH_realAnchor_randomBatch(batch_size=batch_size, class_num=class_num, data_real=X_test_r, labels_real=y_test_r, data_syn=X_test_s, labels_syn=y_test_s)
# val_generator = DataGeneratorBH_realAnchor_randomBatch(batch_size=batch_size, class_num=class_num, data_real=X_val_r, labels_real=y_val_r, data_syn=X_val_s, labels_syn=y_val_s)

# callbacks
history = LossHistory(os.path.join(model_path, f"loss.png"), os.path.join(model_path, f"loss.npy"))
early_stopping = CustomizedEarlyStopping(monitor='val_loss', patience=100, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights.hdf5'), monitor='val_loss', save_best_only=True)
# gradient_logger = GradientExplosionCallback()

history = net.fit(train_generator, steps_per_epoch=len(y_train_s) // (P*K), epochs=epochs,
                      validation_data=val_generator,
                      callbacks=[history, early_stopping, model_checkpoint])

net.save_weights(os.path.join(model_path, "model_weights.hdf5"))


X_train = net.predict(X_train_s[:,:,:,np.newaxis])
y_train = y_train_s


clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)
dump(clf, os.path.join(model_path, 'svm.joblib'))
# clf = load(os.path.join(model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(f"train accuracy: {accuracy_score(y_train, pre_train)}")

X_test = net.predict(X_test_r[:,:,:,np.newaxis])
y_test = y_test_r

pre = clf.predict(X_test)
print(f"test accuracy: {accuracy_score(y_test, pre)}")
# print(recall_score(y_test, pre, average="weighted"))

matrix = confusion_matrix(y_test, pre)
# sns.heatmap(matrix)
# plt.show()
print(matrix.diagonal()/matrix.sum(axis=1))


# history = net.fit(train_generator, steps_per_epoch=5, epochs=epochs,
#                       validation_data=val_generator,
#                       callbacks=[history, early_stopping, model_checkpoint])


# while batch_size >= 2:
#     batch_size = batch_size // 2
#     print("batch size: {}".format(batch_size))
#     history = LossHistory(os.path.join(model_path, f"loss_{batch_size}.png"), os.path.join(model_path, f"loss_{batch_size}.npy"))
#     model_checkpoint = ModelCheckpoint(os.path.join(model_path, f'model_weights_{batch_size}.hdf5'), monitor='val_loss', save_best_only=True)
#     history = net.fit(train_generator, steps_per_epoch=triplet_info["train"]["shape"][0] // batch_size, epochs=epochs,
#                       validation_data=val_generator,
#                       callbacks=[history, early_stopping, model_checkpoint])


