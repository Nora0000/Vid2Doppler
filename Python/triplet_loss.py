import torch
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.layers import Dropout, Conv2DTranspose, Activation
from tensorflow.keras import activations, Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.losses import mae
import tensorflow as tf
import os
import numpy as np
import argparse
from helper import root_mean_squared_error, get_spectograms
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import h5py
from prepare_data import augment_data
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.python.keras.models import Model
from prepare_data import interpolate
from torch.nn import UpsamplingBilinear2d
import math
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.ndimage import gaussian_filter1d

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


