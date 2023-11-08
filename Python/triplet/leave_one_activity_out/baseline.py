import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

leave_out = 5
path1 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x1 = np.load(os.path.join(path1, "X_4.npy"))
y1 = np.load(os.path.join(path1, "Y_4.npy"))
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, random_state=42)

path2 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x2 = np.load(os.path.join(path2, "X_4.npy"))
y2 = np.load(os.path.join(path2, "Y_4.npy"))
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, random_state=42)

X_train = np.vstack((x1_train, x2_train))
y_train = np.concatenate((y1_train, y2_train))

X_train = np.reshape(X_train, [-1, 28*52])

indices = (y_train != leave_out)
X_train = X_train[indices]
y_train = y_train[indices]

clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)


pre_train = clf.predict(X_train)
print(f"train acc: {accuracy_score(y_train, pre_train)}")

X_test = np.vstack((x1_test, x2_test))
X_test = np.reshape(X_test, [-1, 28*52])
y_test = np.concatenate((y1_test, y2_test))

pre_test = clf.predict(X_test)
# print(f"test acc overall: {f1_score(y_test, pre_test, average='weighted')}")
print(f"test acc overall: {accuracy_score(y_test, pre_test)}")

indices = (y_test != leave_out)
X_test1 = X_test[indices]
y_test1 = y_test[indices]
pre_test = clf.predict(X_test1)
# print(f"test acc of known act: {f1_score(y_test1, pre_test, average='weighted')}")
print(f"test acc of known act: {accuracy_score(y_test1, pre_test)}")

indices = (y_test == leave_out)
X_test1 = X_test[indices]
y_test1 = y_test[indices]
pre_test = clf.predict(X_test1)
# print(f"test acc of leave out act: {f1_score(y_test1, pre_test, average='weighted')}")
print(f"test acc of leave out act: {accuracy_score(y_test1, pre_test)}")




