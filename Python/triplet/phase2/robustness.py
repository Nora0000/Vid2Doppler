import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from triplet.utils import triplet_loss
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


emb_size = 128

net = load_model(emb_model_path + "model_weights.hdf5", custom_objects={'triplet_loss': triplet_loss})
embed_model = Model(
  inputs=net.layers[3].input,
  outputs=net.layers[3].output
)


# training data is used to train embed model.
# synthetic test data is used to train classification model.
# real test data is used to test classification model.
X_real = np.load(os.path.join(emb_model_path, "X.npy"))
y_real = np.load(os.path.join(emb_model_path, "Y.npy"))
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

X_syn = np.load(os.path.join(emb_model_path, "X_syn.npy"))
y_syn = np.load(os.path.join(emb_model_path, "Y_syn.npy"))
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)


# data from unseen viewpoint / class
def load_additional_data(real_data_path, real_label_path, syn_data_path, syn_label_path):

    X_real = np.load(real_data_path)
    y_real = np.load(real_label_path)
    X_syn = np.load(syn_data_path)
    y_syn = np.load(syn_label_path)


    X_real = np.delete(X_real, [14, 15, 16, 17], axis=1)
    X_real = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_real[np.newaxis, :, :, :])).numpy()[0, :, :, :]
    X_syn = np.delete(X_syn, [14, 15, 16, 17], axis=1)
    X_syn = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_syn[np.newaxis, :, :, :])).numpy()[0, :, :, :]


    emb_real = embed_model.predict(X_real)
    # add same proportion of data from new viewpoint into train and test dataset
    # ind = random.sample(np.arange(0, len(X_push_v5_r)).tolist(), int((len(X_push_v5_s)/len(X_test_s)) * len(len(X_test_r))))
    # push_emd_r = push_emd_r[ind, :]
    # y_push_v5_r = y_push_v5_r[ind]
    emb_syn = embed_model.predict(X_syn)
    return emb_real, y_real, emb_syn, y_syn


# unseen viewpoint. load "push" at viewpoint 5
real_data_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/X_push.npy"
real_label_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/Y_push.npy"
syn_data_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/X_push_syn.npy"
syn_label_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/Y_push_syn.npy"
# # unseen class. load "bend" at viewpoint 6
# real_data_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/X_bend.npy"
# real_label_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/Y_bend.npy"
# syn_data_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/X_bend_syn.npy"
# syn_label_path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/Y_bend_syn.npy"
emb_real, y_real, emb_syn, y_syn = load_additional_data(real_data_path, real_label_path, syn_data_path, syn_label_path)

emb_size = 128

X_train = embed_model.predict(X_train_s)
X_train = np.vstack((X_train, emb_syn))
y_train = np.concatenate((y_train_s, y_syn))


clf = svm.SVC(C=1000, kernel="rbf")
clf.fit(X_train, y_train)
dump(clf, os.path.join(emb_model_path, 'svm_new_viewpoint.joblib'))

pre_train = clf.predict(X_train)
print(accuracy_score(y_train, pre_train))

X_test = emb_model_path.predict(X_test_r)
X_test = np.vstack((X_test, emb_real))
y_test = np.concatenate((y_test_r, y_real))

pre = clf.predict(X_test)
print(accuracy_score(y_test, pre))
