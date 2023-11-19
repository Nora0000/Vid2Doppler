"""
in training phase 2, train with 3 views' synthetic data,
test with 3 views' real data.
the embedding model is trained with 2 views' real and synthetic data, one view's synthetic data

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
from tensorflow.keras.models import Model
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_path = "../../../models/triplet/"
model_path = "/home/mengjingliu/Vid2Doppler/models/triplet_cross_view_real0.08/with_synthetic_of_unseen_view/triplet_v46_5/"
if not os.path.exists(model_path):
    print("{} does not exist.".format(model_path))
    os.mkdir(model_path)


emb_size = 128

net = load_model(model_path + "model_weights_32.hdf5", custom_objects={'triplet_loss': triplet_loss})
embed_model = Model(
  inputs=net.layers[3].input,
  outputs=net.layers[3].output
)

X_train_s = np.load(os.path.join(model_path, "X_train_syn.npy"))
X_test_r = np.load(os.path.join(model_path, "X_test_real.npy"))
y_train_s = np.load(os.path.join(model_path, "Y_train_syn.npy"))
y_test_r = np.load(os.path.join(model_path, "Y_test_real.npy"))
X_train_r = np.load(os.path.join(model_path, "X_train_real.npy"))
y_train_r = np.load(os.path.join(model_path, "Y_train_real.npy"))

emb_size = 128


X_train_s = embed_model.predict(X_train_s)
X_train_r = embed_model.predict(X_train_r)
X_train = np.vstack((X_train_s, X_train_r))
y_train = np.concatenate((y_train_s, y_train_r))

# clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
clf = svm.SVC(C=1, kernel="rbf")
# clf = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=100000, random_state=42)
clf.fit(X_train, y_train)
# dump(clf, os.path.join(model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(f"training accuracy: {accuracy_score(y_train, pre_train)}")

X_test = embed_model.predict(X_test_r)
pre_v46 = clf.predict(X_test)
print(f"two seen views' accuracy: {accuracy_score(y_test_r, pre_v46)}")

Y_test_real_leaveout = np.load(os.path.join(model_path, "Y_test_real_leaveout.npy"))
X_test_real_leaveout = np.load(os.path.join(model_path, "X_test_real_leaveout.npy"))
pre_leaveout = clf.predict(embed_model.predict(X_test_real_leaveout))
print(f"unseen view's accuracy: {accuracy_score(Y_test_real_leaveout, pre_leaveout)}")


pre = np.concatenate((pre_v46, pre_leaveout))
y_test = np.concatenate((y_test_r, Y_test_real_leaveout))

print(f"testing accuracy: {accuracy_score(y_test, pre)}")

# matrix = confusion_matrix(y_test, pre)
# sns.heatmap(matrix)
# plt.ylabel("prediction")
# plt.xlabel("label")
# plt.show()
# print(matrix.diagonal()/matrix.sum(axis=1))

