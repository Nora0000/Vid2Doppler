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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_path = "../../../models/triplet/"
model_path = "../../../models/triplet_v46_5/"
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
# views = ['4', '5', '6']
# newview = '5'
# X_real = []
# for v in views:
# 	path_v = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR{v}"
# 	if len(X_real) == 0:
# 		X_real = np.load(os.path.join(path_v, "X_4.npy"))
# 		y_real = np.load(os.path.join(path_v, "Y_4.npy")) - 1
# 		X_syn = np.load(os.path.join(path_v, "X_4_syn.npy"))
# 		y_syn = np.load(os.path.join(path_v, "Y_4_syn.npy")) - 1
# 	else:
# 		X_real = np.vstack((np.load(os.path.join(path_v, "X_4.npy")), X_real))
# 		y_real = np.concatenate((np.load(os.path.join(path_v, "Y_4.npy")) - 1, y_real))
# 		X_syn = np.vstack((np.load(os.path.join(path_v, "X_4_syn.npy")), X_syn))
# 		y_syn = np.concatenate((np.load(os.path.join(path_v, "Y_4_syn.npy")) - 1, y_syn))
#
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)
# X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

# path_v_new = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR{newview}"
# X_real_new = np.load(os.path.join(path_v_new, "X_4.npy"))
# y_real_new = np.load(os.path.join(path_v_new, "Y_4.npy")) - 1
# X_syn_new = np.load(os.path.join(path_v_new, "X_4_syn.npy"))
# y_syn_new = np.load(os.path.join(path_v_new, "Y_4_syn.npy")) - 1
#
# _, X_test_r_new, _, y_test_r_new = train_test_split(X_real_new, y_real_new, test_size=0.2, random_state=42)
# _, X_train_s_new, _, y_train_s_new = train_test_split(X_syn_new, y_syn_new, test_size=0.8, random_state=42)

# emb_real_new = embed_model.predict(X_test_r_new)
# print(f"testing data size: {len(X_test_r)}, {len(emb_real_new)}, {len(X_test_r)+len(emb_real_new)}")

emb_size = 128


X_train = embed_model.predict(X_train_s)
y_train = y_train_s

# clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
clf = svm.SVC(C=1, kernel="rbf")
clf.fit(X_train, y_train)
dump(clf, os.path.join(model_path, 'svm.joblib'))

pre_train = clf.predict(X_train)
print(f"training accuracy: {accuracy_score(y_train, pre_train)}")

X_test = embed_model.predict(X_test_r)
pre_v46 = clf.predict(X_test)
print(f"v46 accuracy: {accuracy_score(y_test_r, pre_v46)}")

Y_test_real_leaveout = np.load(os.path.join(model_path, "Y_test_real_leaveout.npy"))
X_test_real_leaveout = np.load(os.path.join(model_path, "X_test_real_leaveout.npy"))
pre_leaveout = clf.predict(embed_model.predict(X_test_real_leaveout))
print(f"v5 accuracy: {accuracy_score(Y_test_real_leaveout, pre_leaveout)}")


pre = np.concatenate((pre_v46, pre_leaveout))
y_test = np.concatenate((y_test_r, Y_test_real_leaveout))

print(f"testing accuracy: {accuracy_score(y_test, pre)}")

# matrix = confusion_matrix(y_test, pre)
# sns.heatmap(matrix)
# plt.ylabel("prediction")
# plt.xlabel("label")
# plt.show()
# print(matrix.diagonal()/matrix.sum(axis=1))



# p = (y_test != 6)
# Xt = X_test[p]
# yt = y_test[p]
#
# pre = clf.predict(Xt)
# print(accuracy_score(yt, pre))
#
# p = (y_test == 6)
# Xt = X_test[p]
# yt = y_test[p]
#
# pre = clf.predict(Xt)
# print(accuracy_score(yt, pre))
