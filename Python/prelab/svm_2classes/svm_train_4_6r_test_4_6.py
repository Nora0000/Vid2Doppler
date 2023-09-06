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
N = 2

new_model_path = "../../../models/svm_train_4_6r_1%_test_4_6/"
if not os.path.exists(new_model_path):
	os.mkdir(new_model_path)

# Load the data and split it into train and test sets
path4 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x4 = np.load(os.path.join(path4, "X_2.npy"))
y4 = np.load(os.path.join(path4, "Y_2.npy")) - 1
# x4_aug, y4_aug = augment_data(x4, y4)
# X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(x4_aug, y4_aug, test_size=0.2, random_state=42)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(x4, y4, test_size=0.2, random_state=42)
X_train_4_aug, y_train_4_aug = augment_data(X_train_4, y_train_4)

# sensor 6 real data for test
# sensor 6 real data for test
path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x6_real = np.load(os.path.join(path6, "X_2.npy"))
y6_real = np.load(os.path.join(path6, "Y_2.npy")) - 1
# augment before split
# x6_aug, y6_aug = augment_data(x6_real, y6_real)
# x6r, X_test_6, y6r, y_test_6 = train_test_split(x6_aug, y6_aug, test_size=0.2, random_state=42)
# X_train_6_real, _, y_train_6_real, _ = train_test_split(x6r, y6r, test_size=0.99, random_state=42)
# augment after split
x6r, X_test_6, y6r, y_test_6 = train_test_split(x6_real, y6_real, test_size=0.2, random_state=42)
selected = []
for i in range(N):
	selected_t = np.random.choice(np.where(y6r==i)[0], 1)
	selected.extend(selected_t)
x6_real_selected = x6r[selected, :, :]
y6_real_selected = np.arange(0, N)
# x6_real_selected, y6_real_selected = augment_data(x6_real_selected, y6_real_selected)



# X_train = X_train_4
# y_train = y_train_4
# X_test = X_test_4
# y_test = y_test_4
X_train = np.vstack((X_train_4_aug, x6_real_selected))
y_train = np.concatenate((y_train_4_aug, y6_real_selected))
X_test = np.vstack((X_test_4, X_test_6))
y_test = np.concatenate((y_test_4, y_test_6))

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

# np.save(os.path.join(new_model_path, "X_test.npy"), X_test)
# np.save(os.path.join(new_model_path, "y_test.npy"), y_test)

clf = svm.SVC(C=10, kernel="poly")
clf.fit(X_train, y_train)
# dump(clf, os.path.join(new_model_path, 'svm.joblib'))
# clf = load(os.path.join(new_model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(accuracy_score(y_train, pre_train))

pre = clf.predict(X_test)
print(accuracy_score(y_test, pre))
# print(recall_score(y_test, pre, average="weighted"))
# print(f1_score(y_test, pre, average="weighted"))

