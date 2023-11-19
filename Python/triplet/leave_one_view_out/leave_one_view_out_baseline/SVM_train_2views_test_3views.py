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

data_size = 0.2


path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
x1 = np.load(os.path.join(path1, "X_4.npy"))
y1 = np.load(os.path.join(path1, "Y_4.npy"))
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, random_state=42)
x1_train, _, y1_train, _ = train_test_split(x1_train, y1_train, train_size=data_size, random_state=42)

path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x2 = np.load(os.path.join(path2, "X_4.npy"))
y2 = np.load(os.path.join(path2, "Y_4.npy"))
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, random_state=42)
x2_train, _, y2_train, _ = train_test_split(x2_train, y2_train, train_size=data_size, random_state=42)


path3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x3 = np.load(os.path.join(path3, "X_4.npy"))
y3 = np.load(os.path.join(path3, "Y_4.npy"))
_, x3_test, _, y3_test = train_test_split(x3, y3, test_size=0.2, random_state=42)


X_train = np.vstack((x1_train, x2_train))
y_train = np.concatenate((y1_train, y2_train))

X_train = np.reshape(X_train, [-1, 28*52])

clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=10000, random_state=42)
# clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)


pre_train = clf.predict(X_train)
print(f"train acc: {accuracy_score(y_train, pre_train)}")

X_test_2view = np.vstack((x1_test, x2_test))
X_test_2view = np.reshape(X_test_2view, [-1, 28*52])
y_test_2view = np.concatenate((y1_test, y2_test))
pre_test_2view = clf.predict(X_test_2view)
print(f"test acc at 2 known views: {accuracy_score(y_test_2view, pre_test_2view)}")

x3_test = np.reshape(x3_test, [-1, 28*52])
pre_v3 = clf.predict(x3_test)
print(f"test acc at unseen view: {accuracy_score(y3_test, pre_v3)}")

y_test = np.concatenate((y_test_2view, y3_test))
pre = np.concatenate((pre_test_2view, pre_v3))
print(f"test acc overall: {accuracy_score(y_test, pre)}")




