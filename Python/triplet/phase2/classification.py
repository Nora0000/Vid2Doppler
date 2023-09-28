import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from ..utils import triplet_loss
from torch.nn import UpsamplingBilinear2d
import torch
import random
from tensorflow.keras.models import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# clf_model_path = "../../models/classification/"
emb_model_path = "../../../models/triplet/"
if not os.path.exists(emb_model_path):
    print("{} does not exist.".format(emb_model_path))
    os.mkdir(emb_model_path)



net = load_model(emb_model_path + "model_weights.hdf5", custom_objects={'triplet_loss': triplet_loss})
embed_model = Model(
  inputs=net.layers[3].input,
  outputs=net.layers[3].output
)


X_real = np.load(os.path.join(emb_model_path, "X.npy"))
y_real = np.load(os.path.join(emb_model_path, "Y.npy"))
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
# X_train_r_aug, y_train_r_aug = augment_data(X_train_r, y_train_r)

X_syn = np.load(os.path.join(emb_model_path, "X_syn.npy"))
y_syn = np.load(os.path.join(emb_model_path, "Y_syn.npy"))
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)

emb_size = 128

X_train = embed_model.predict(X_train_s)
y_train = y_train_s


clf = svm.SVC(C=1000, kernel="rbf")
clf.fit(X_train, y_train)
dump(clf, os.path.join(emb_model_path, 'svm.joblib'))
# clf = load(os.path.join(new_model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(accuracy_score(y_train, pre_train))

X_test = embed_model.predict(X_test_s)
y_test = y_test_r

pre = clf.predict(X_test)
print(accuracy_score(y_test, pre))
