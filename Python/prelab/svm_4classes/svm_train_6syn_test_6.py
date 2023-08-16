import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Load the Fashion MNIST dataset
# from keras.datasets import fashion_mnist

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

new_model_path = "../../../models/svm_2classes_train_6syn_test_6/"
if not os.path.exists(new_model_path):
	os.mkdir(new_model_path)

# sensor 6 synthetic data for training
path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x6_syn = np.load(os.path.join(path6, "X_4_syn.npy"))
y6_syn = np.load(os.path.join(path6, "Y_4_syn.npy")) - 1
x6_aug_syn, y6_aug_syn = augment_data(x6_syn, y6_syn)

X_train = x6_aug_syn
y_train = y6_aug_syn

# sensor 6 real data for test
path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x6_real = np.load(os.path.join(path6, "X_4.npy"))
y6_real = np.load(os.path.join(path6, "Y_4.npy")) - 1
# x6_aug_real, y6_aug_real = augment_data(x6_real, y6_real)
# _, X_test_6, _, y_test_6 = train_test_split(x6_aug_real, y6_aug_real, test_size=0.2, random_state=42)

X_test = x6_real
y_test = y6_real

# Normalize the pixel values to range [0, 1]
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

# Convert class labels to one-hot encoded vectors
# num_classes = 5
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# Reshape the data to fit the CNN input shape
X_train = np.reshape(X_train, [-1, 32*50])
X_test = np.reshape(X_test, [-1, 32*50])

np.save(os.path.join(new_model_path, "X_test.npy"), X_test)
np.save(os.path.join(new_model_path, "y_test.npy"), y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

clf = svm.SVC(C=100, kernel="poly")
clf.fit(X_train, y_train)
dump(clf, os.path.join(new_model_path, 'svm.joblib'))
# clf = load(os.path.join(new_model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(accuracy_score(y_train, pre_train))

pre_val = clf.predict(X_val)
print(accuracy_score(y_val, pre_val))

pre = clf.predict(X_test)
print(accuracy_score(y_test, pre))
# print(recall_score(y_test, pre, average="weighted"))
# print(f1_score(y_test, pre, average="weighted"))

