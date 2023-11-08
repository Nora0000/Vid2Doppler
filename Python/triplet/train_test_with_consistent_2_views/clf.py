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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_path = "../../../models/triplet/"
model_path = "../../../models/triplet_v56/"
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
X_train_r = np.load(os.path.join(model_path, "X_train_real.npy"))
X_test_r = np.load(os.path.join(model_path, "X_test_real.npy"))
y_train_s = np.load(os.path.join(model_path, "Y_train_syn.npy"))
y_train_r = np.load(os.path.join(model_path, "Y_train_real.npy"))
y_test_r = np.load(os.path.join(model_path, "Y_test_real.npy"))

# y_real_new = np.load(os.path.join(path_v3, "Y_4.npy")) - 1

emb_size = 128

X_train = np.vstack((X_train_r, X_train_s))
X_train = embed_model.predict(X_train)
y_train = np.concatenate((y_train_r, y_train_s))

# clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
clf = svm.SVC(C=10, kernel="rbf")
clf.fit(X_train, y_train)
dump(clf, os.path.join(model_path, 'svm_train2_test2_real_in_phase2.joblib'))

pre_train = clf.predict(X_train)
print(f"training accuracy: {accuracy_score(y_train, pre_train)}")

X_test = embed_model.predict(X_test_r)
pre_v46 = clf.predict(X_test)
print(f"test accuracy: {accuracy_score(y_test_r, pre_v46)}")
