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
from scipy.ndimage import gaussian_filter1d


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
loss = np.load("/home/mengjingliu/Vid2Doppler/models/triplet_v46_test/loss_64.npy")
train_loss, val_loss = loss[0, 1:], loss[1, 1:]

loss = np.load("/home/mengjingliu/Vid2Doppler/models/triplet_v46_test/loss_32.npy")
train_loss = np.concatenate((train_loss, loss[0, :]))
val_loss = np.concatenate((val_loss, loss[1, :]))

loss = np.load("/home/mengjingliu/Vid2Doppler/models/triplet_v46_test/loss_16.npy")
train_loss = np.concatenate((train_loss, loss[0, :]))
val_loss = np.concatenate((val_loss, loss[1, :]))

loss = np.load("/home/mengjingliu/Vid2Doppler/models/triplet_v46_test/loss_8.npy")
train_loss = np.concatenate((train_loss, loss[0, :]))
val_loss = np.concatenate((val_loss, loss[1, :]))

plt.plot(np.arange(0, len(train_loss)), train_loss)
plt.plot(np.arange(0, len(val_loss)), val_loss)
plt.show()
