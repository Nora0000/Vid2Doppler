"""
train with two views' real and synthetic data, 80%.
tets with three views' real data, 20%.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from triplet.utils import triplet_loss
from torch.nn import UpsamplingBilinear2d
import torch
import random
from sklearn.neural_network import MLPClassifier


real_data_size = 0.2

path_v1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
X_real = np.load(os.path.join(path_v1, "X_4.npy"))
y_real = np.load(os.path.join(path_v1, "Y_4.npy")) - 1
X_syn = np.load(os.path.join(path_v1, "X_4_syn.npy"))
y_syn = np.load(os.path.join(path_v1, "Y_4_syn.npy")) - 1

path_v2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
X_real = np.vstack((np.load(os.path.join(path_v2, "X_4.npy")), X_real))
y_real = np.concatenate((np.load(os.path.join(path_v2, "Y_4.npy")) - 1, y_real))
X_syn = np.vstack((np.load(os.path.join(path_v2, "X_4_syn.npy")), X_syn))
y_syn = np.concatenate((np.load(os.path.join(path_v2, "Y_4_syn.npy")) - 1, y_syn))

path_v3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
X_syn = np.vstack((np.load(os.path.join(path_v3, "X_4_syn.npy")), X_syn))
y_syn = np.concatenate((np.load(os.path.join(path_v3, "Y_4_syn.npy")) - 1, y_syn))


# X_real = (X_real - np.mean(X_real)) / np.std(X_real)
# X_syn = (X_syn - np.mean(X_syn)) / np.std(X_syn)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
X_train_r, _, y_train_r, _ = train_test_split(X_train_r, y_train_r, train_size=real_data_size, random_state=42)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)


x3 = np.load(os.path.join(path_v3, "X_4.npy"))
y3 = np.load(os.path.join(path_v3, "Y_4.npy"))
_, x3_test, _, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)

X_train = np.vstack((X_train_r, X_train_s))
y_train = np.concatenate((y_train_r, y_train_s))

X_train = np.reshape(X_train, [-1, 28*52])

clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=10000, random_state=42)
# clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)


pre_train = clf.predict(X_train)
print(f"train acc: {accuracy_score(y_train, pre_train)}")

X_test_2view = X_test_r
X_test_2view = np.reshape(X_test_2view, [-1, 28*52])
y_test_2view = y_test_r
pre_test_2view = clf.predict(X_test_2view)
print(f"test acc at 2 known views: {accuracy_score(y_test_2view, pre_test_2view)}")

x3_test = np.reshape(x3_test, [-1, 28*52])
pre_v3 = clf.predict(x3_test)
print(f"test acc at unseen view: {accuracy_score(y3_test, pre_v3)}")

y_test = np.concatenate((y_test_2view, y3_test))
pre = np.concatenate((pre_test_2view, pre_v3))
print(f"test acc overall: {accuracy_score(y_test, pre)}")




