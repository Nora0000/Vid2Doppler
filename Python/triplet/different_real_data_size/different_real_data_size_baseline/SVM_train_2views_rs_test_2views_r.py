"""
train with two views' real and synthetic data, 80%.
tets with two views' real data, 20%.


"""


import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

real_data_size = 0.1

path0 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x0 = np.load(os.path.join(path0, "X_4.npy"))
y0 = np.load(os.path.join(path0, "Y_4.npy"))
x0_train, x0_test, y0_train, y0_test = train_test_split(x0, y0, train_size=0.8, random_state=42)
x0_train, _, y0_train, _ = train_test_split(x0_train, y0_train, train_size=real_data_size, random_state=42)
x0_syn = np.load(os.path.join(path0, "X_4_syn.npy"))
y0_syn = np.load(os.path.join(path0, "Y_4_syn.npy"))
x0_syn, _, y0_syn, _ = train_test_split(x0_syn, y0_syn, train_size=0.8, random_state=42)
x0_syn, _, y0_syn, _ = train_test_split(x0_syn, y0_syn, train_size=real_data_size, random_state=42)
x0_train = np.vstack((x0_train, x0_syn))
y0_train = np.concatenate((y0_train, y0_syn))

path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5"
x1 = np.load(os.path.join(path1, "X_4.npy"))
y1 = np.load(os.path.join(path1, "Y_4.npy"))
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, random_state=42)
x1_train, _, y1_train, _ = train_test_split(x1_train, y1_train, train_size=real_data_size, random_state=42)
x1_syn = np.load(os.path.join(path1, "X_4_syn.npy"))
y1_syn = np.load(os.path.join(path1, "Y_4_syn.npy"))
x1_syn, _, y1_syn, _ = train_test_split(x1_syn, y1_syn, train_size=0.8, random_state=42)
x1_syn, _, y1_syn, _ = train_test_split(x1_syn, y1_syn, train_size=real_data_size, random_state=42)
x1_train = np.vstack((x1_train, x1_syn))
y1_train = np.concatenate((y1_train, y1_syn))

path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x2 = np.load(os.path.join(path2, "X_4.npy"))
y2 = np.load(os.path.join(path2, "Y_4.npy"))
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, random_state=42)
x2_train, _, y2_train, _ = train_test_split(x2_train, y2_train, train_size=real_data_size, random_state=42)
x2_syn = np.load(os.path.join(path2, "X_4_syn.npy"))
y2_syn = np.load(os.path.join(path2, "Y_4_syn.npy"))
x2_syn, _, y2_syn, _ = train_test_split(x2_syn, y2_syn, train_size=0.8, random_state=42)
x2_syn, _, y2_syn, _ = train_test_split(x2_syn, y2_syn, train_size=real_data_size, random_state=42)
x2_train = np.vstack((x2_train, x2_syn))
y2_train = np.concatenate((y2_train, y2_syn))

X_train = np.vstack((np.vstack((x0_train, x1_train)), x2_train))
y_train = np.concatenate((y0_train, np.concatenate((y1_train, y2_train))))

X_train = np.reshape(X_train, [-1, 28*52])
# clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=10000, random_state=42)
# clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=10000, random_state=42)
clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)


pre_train = clf.predict(X_train)
print(f"train acc: {accuracy_score(y_train, pre_train)}")

X_test_2view = np.vstack((x1_test, x2_test))
X_test_2view = np.reshape(X_test_2view, [-1, 28*52])
y_test_2view = np.concatenate((y1_test, y2_test))
pre_test_2view = clf.predict(X_test_2view)
print(f"test acc at 2 known views: {accuracy_score(y_test_2view, pre_test_2view)}")



