import numpy as np
import math
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse
from helper import get_spectograms, color_scale, chose_central_rangebins
import cv2
import matplotlib
from helper import load_txt_to_datetime
from segment import segment
from prepare_data import load_data, augment_data
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, schedules
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from helper import root_mean_squared_error

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# cnn test
model_path = "/home/mengjingliu/Vid2Doppler/models/encoder_s6"
X_train = np.load(os.path.join(model_path, "X_train.npy"))
y_train = np.load(os.path.join(model_path, "y_train.npy"))
# path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
# x6_syn = np.load(os.path.join(path6, "X_4_syn.npy"))

model = load_model(os.path.join(model_path, "autoencoder_weights.hdf5"), custom_objects={'root_mean_squared_error': root_mean_squared_error})

X_test = np.load(os.path.join(model_path, "X_test.npy"))
y_test = np.load(os.path.join(model_path, "y_test.npy"))

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
