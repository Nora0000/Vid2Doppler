"""
train with two views' real data, 80%.
tets with two views' real data, 20%.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x1 = np.load(os.path.join(path1, "X_4.npy"))
y1 = np.load(os.path.join(path1, "Y_4.npy"))

path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
x1 = np.vstack((np.load(os.path.join(path2, "X_4.npy")), x1))
y1 = np.concatenate((np.load(os.path.join(path2, "Y_4.npy")), y1))

path3 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x1 = np.vstack((np.load(os.path.join(path3, "X_4.npy")), x1))
y1 = np.concatenate((np.load(os.path.join(path3, "Y_4.npy")), y1))


X_train, X_test, y_train, y_test = train_test_split(x1, y1, train_size=0.8, random_state=42)
# X_train, _, y_train, _ = train_test_split(x1, y1, train_size=0.8, random_state=42)

X_train = np.reshape(X_train, [-1, 28*52])
clf = svm.SVC(C=10, kernel="rbf")
# clf = MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000, random_state=42)
clf.fit(X_train, y_train)


pre_train = clf.predict(X_train)
print(f"train acc: {accuracy_score(y_train, pre_train)}")

# X_test_2view = np.vstack((x1_test, x2_test))
X_test = np.reshape(X_test, [-1, 28*52])
# y_test_2view = np.concatenate((y1_test, y2_test))
pre_test = clf.predict(X_test)
print(f"test acc at 2 known views: {accuracy_score(y_test, pre_test)}")



