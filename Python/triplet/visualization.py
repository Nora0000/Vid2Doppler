import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from utils import triplet_loss
from torch.nn import UpsamplingBilinear2d
import torch
import random
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

emb_model_path = "../../models/triplet/"
if not os.path.exists(emb_model_path):
    print("{} does not exist.".format(emb_model_path))
    # os.mkdir(emb_model_path)

net = load_model(emb_model_path + "model_weights.hdf5", custom_objects={'triplet_loss': triplet_loss})

# input_shape = (28, 52, 1)
# input = tf.keras.layers.Input(shape=input_shape)

cnn_1 = Model(
  inputs=net.layers[3].layers[0].input,
  outputs=net.layers[3].layers[1].output)

cnn_2 = Model(
  inputs=net.layers[3].layers[0].input,
  outputs=net.layers[3].layers[3].output)

X_real = np.load(os.path.join(emb_model_path, "X.npy"))
y_real = np.load(os.path.join(emb_model_path, "Y.npy"))

X_syn = np.load(os.path.join(emb_model_path, "X_syn.npy"))
y_syn = np.load(os.path.join(emb_model_path, "Y_syn.npy"))

pre1_r = cnn_1.predict([X_real])
pre1_s = cnn_1.predict([X_syn])

pre2_r = cnn_2.predict([X_real])
pre2_s = cnn_2.predict([X_syn])

for pr, ps in zip(pre1_r[160:], pre1_s[160:]):

    sns.heatmap(pr[:, :, 0])
    plt.show()
    sns.heatmap(ps[:, :, 0])
    plt.show()
