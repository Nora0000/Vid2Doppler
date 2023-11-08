import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from triplet.utils import triplet_loss, triplet_loss_bh
# from triplet.different_triplet_loss.utils import triple_loss_bh
from torch.nn import UpsamplingBilinear2d
import torch
import random
from tensorflow.keras.models import Model
import seaborn as sns
from matplotlib import pyplot as plt
from triplet.different_triplet_loss_single_input.embedding_model import EmbeddingModel
from tensorflow.keras.layers import LeakyReLU
# from triplet.different_triplet_loss.utils import triplet_loss_bh
from triplet.different_triplet_loss.triplet_loss_batch_hard.utils import triplet_loss_bh
from sklearn.neural_network import MLPClassifier

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# data_path = "../../../models/triplet/"
emb_model_path = "/home/mengjingliu/Vid2Doppler/models/triplet_v46_test/"
if not os.path.exists(emb_model_path):
    print("{} does not exist.".format(emb_model_path))
    os.mkdir(emb_model_path)


# from triplet.different_triplet_loss.embedding_model import EmbeddingModel

# net = EmbeddingModel()
# net.build(input_shape=(None, 28, 52, 1))
# net.load_weights(emb_model_path + "model_weights.hdf5")
# embed_model = net
net = load_model(emb_model_path + "model_weights_64.hdf5", custom_objects={'triplet_loss': triplet_loss})
# embed_model = Model(
#   inputs=net.layers[2].input,
#   outputs=net.layers[2].output
# )
embed_model = Model(
  inputs=net.layers[3].input,
  outputs=net.layers[3].output
)


X_train_s = np.load(os.path.join(emb_model_path, "X_train_syn.npy"))[:, :, :, np.newaxis]
y_train_s = np.load(os.path.join(emb_model_path, "Y_train_syn.npy"))
X_test_r = np.load(os.path.join(emb_model_path, "X_test_real.npy"))[:, :, :, np.newaxis]
y_test_r = np.load(os.path.join(emb_model_path, "Y_test_real.npy"))

# X_real = np.load(os.path.join(data_path, "X.npy"))
# y_real = np.load(os.path.join(data_path, "Y.npy")) - 1
# X_syn = np.load(os.path.join(data_path, "X_syn.npy"))
# y_syn = np.load(os.path.join(data_path, "Y_syn.npy")) - 1

# # add v5
# activities = ["push", "circle", "step", "stand", "sit"]
# path_v5 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
# for act in activities:
#     X_real_a = np.load(os.path.join(path_v5, f"X_{act}.npy"))
#     y_real_a = np.load(os.path.join(path_v5, f"Y_{act}.npy")) - 1
#     X_syn_a = np.load(os.path.join(path_v5, f"X_{act}_syn.npy"))
#     y_syn_a = np.load(os.path.join(path_v5, f"Y_{act}_syn.npy")) - 1
#
#     X_real_a = np.delete(X_real_a, [14, 15, 16, 17], axis=1)
#     X_real_a = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_real_a[np.newaxis, :, :, :])).numpy()[0, :, :, :]
#     X_syn_a = np.delete(X_syn_a, [14, 15, 16, 17], axis=1)
#     X_syn_a = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_syn_a[np.newaxis, :, :, :])).numpy()[0, :, :, :]
#
#     X_real = np.vstack((X_real, X_real_a))
#     X_syn = np.vstack((X_syn, X_syn_a))
#     y_real = np.concatenate((y_real, y_real_a))
#     y_syn = np.concatenate((y_syn, y_syn_a))

# X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)

emb_size = 128

X_train = embed_model.predict(X_train_s)
y_train = y_train_s

# clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=10000, random_state=42)
# clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=100000, random_state=42)
clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)
# dump(clf, os.path.join(emb_model_path, 'svm.joblib'))
# clf = load(os.path.join(emb_model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(f"train accuracy: {accuracy_score(y_train, pre_train)}")

X_test = embed_model.predict(X_test_r)
y_test = y_test_r

pre = clf.predict(X_test)
print(f"test accuracy: {accuracy_score(y_test, pre)}")
# print(recall_score(y_test, pre, average="weighted"))

matrix = confusion_matrix(y_test, pre)
# sns.heatmap(matrix)
# plt.show()
print(matrix.diagonal()/matrix.sum(axis=1))
