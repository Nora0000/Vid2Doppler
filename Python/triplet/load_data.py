"""
load data at multiple viewpoints, concatenate and save them in one file.
preprocess data: delete central lines, upsample to expected dimensions
"""

import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from prepare_data import augment_data
from torch.nn import UpsamplingBilinear2d
from scipy.ndimage import gaussian_filter1d

act = ["circle", "sit", "stand", "step"]
a = "push"
real_data_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/X_{a}.npy"
real_label_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/Y_{a}.npy"
syn_data_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/X_{a}_syn.npy"
syn_label_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/Y_{a}_syn.npy"
X_real = np.load(real_data_path)
y_real = np.load(real_label_path)
X_syn = np.load(syn_data_path)
y_syn = np.load(syn_label_path)

X_real = np.delete(X_real, [14, 15, 16, 17], axis=1)
X_real = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_real[np.newaxis, :, :, :])).numpy()[0, :, :, :]
X_syn = np.delete(X_syn, [14, 15, 16, 17], axis=1)
X_syn = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_syn[np.newaxis, :, :, :])).numpy()[0, :, :, :]

for a in act:
    real_data_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/X_{a}.npy"
    real_label_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/Y_{a}.npy"
    syn_data_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/X_{a}_syn.npy"
    syn_label_path = f"/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/Y_{a}_syn.npy"
    
    X_real_ = np.load(real_data_path)
    y_real_ = np.load(real_label_path)
    X_syn_ = np.load(syn_data_path)
    y_syn_ = np.load(syn_label_path)
    
    X_real_ = np.delete(X_real_, [14, 15, 16, 17], axis=1)
    X_real_ = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_real_[np.newaxis, :, :, :])).numpy()[0, :, :, :]
    X_syn_ = np.delete(X_syn_, [14, 15, 16, 17], axis=1)
    X_syn_ = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_syn_[np.newaxis, :, :, :])).numpy()[0, :, :, :]
    
    X_real = np.vstack((X_real, X_real_))
    X_syn = np.vstack((X_syn, X_syn_))
    y_real = np.concatenate((y_real, y_real_))
    y_syn = np.concatenate((y_syn, y_syn_))

np.save(os.path.join("/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5", "X_4.npy"), X_real)
np.save(os.path.join("/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5", "Y_4.npy"), y_real)
np.save(os.path.join("/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5", "X_4_syn.npy"), X_syn)
np.save(os.path.join("/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5", "Y_4_syn.npy"), y_syn)

exit(0)

path4 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR4"
x4 = np.load(os.path.join(path4, "X_4.npy"))
y4 = np.load(os.path.join(path4, "Y_4.npy"))
x4_syn = np.load(os.path.join(path4, "X_4_syn.npy"))
y4_syn = np.load(os.path.join(path4, "Y_4_syn.npy"))

x4 = np.delete(x4, [14, 15, 16, 17], axis=1)
x4 = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(x4[np.newaxis, :, :, :])).numpy()[0, :, :, :]
x4_syn = np.delete(x4_syn, [14, 15, 16, 17], axis=1)
x4_syn = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(x4_syn[np.newaxis, :, :, :])).numpy()[0, :, :, :]

np.save(os.path.join(path4, "X_4.npy"), x4)
np.save(os.path.join(path4, "Y_4.npy"), y4)
np.save(os.path.join(path4, "X_4_syn.npy"), x4_syn)
np.save(os.path.join(path4, "Y_4_syn.npy"), y4_syn)

path6 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
x6 = np.load(os.path.join(path6, "X_4.npy"))
y6 = np.load(os.path.join(path6, "Y_4.npy"))
x6_syn = np.load(os.path.join(path6, "X_4_syn.npy"))
y6_syn = np.load(os.path.join(path6, "Y_4_syn.npy"))
x6 = np.delete(x6, [14, 15, 16, 17], axis=1)
x6 = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(x6[np.newaxis, :, :, :])).numpy()[0, :, :, :]
x6_syn = np.delete(x6_syn, [14, 15, 16, 17], axis=1)
x6_syn = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(x6_syn[np.newaxis, :, :, :])).numpy()[0, :, :, :]

np.save(os.path.join(path6, "X_4.npy"), x6)
np.save(os.path.join(path6, "Y_4.npy"), y6)
np.save(os.path.join(path6, "X_4_syn.npy"), x6_syn)
np.save(os.path.join(path6, "Y_4_syn.npy"), y6_syn)

# X = np.vstack((x4, x6))
# X_syn = np.vstack((x4_syn, x6_syn))
# y = np.concatenate((y4, y6))
# y_syn = np.concatenate((y4_syn, y6_syn))

# X_aug, _ = augment_data(X, [1] * len(X))
# y_aug, _ = augment_data(y, [1] * len(y))
# X_aug = UpsamplingBilinear2d(size=(32, 52))(torch.tensor(X_aug[np.newaxis, :, :, :])).numpy()[0, :, :, :]
# y_aug = UpsamplingBilinear2d(size=(32, 52))(torch.tensor(y_aug[np.newaxis, :, :, :])).numpy()[0, :, :, :]
# X_aug = np.delete(X_aug, [14, 15, 16, 17], axis=1)
# y_aug = np.delete(y_aug, [14, 15, 16, 17], axis=1)
# for i in range(X_aug.shape[0]):
# 	for j in range(52):
# 		X_aug[i, :, j] = gaussian_filter1d(X_aug[i, :, j], 3)
# X_aug = (X_aug - np.min(X_aug)) / (np.max(X_aug) - np.min(X_aug))
# y_aug = (y_aug - np.min(y_aug)) / (np.max(y_aug) - np.min(y_aug))
# X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# model_path = "../../models/triplet/"
# if not os.path.exists(model_path):
# 	os.mkdir(model_path)
#
# X = np.delete(X, [14, 15, 16, 17], axis=1)
# X = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X[np.newaxis, :, :, :])).numpy()[0, :, :, :]
# X_syn = np.delete(X_syn, [14, 15, 16, 17], axis=1)
# X_syn = UpsamplingBilinear2d(size=(28, 52))(torch.tensor(X_syn[np.newaxis, :, :, :])).numpy()[0, :, :, :]
#
# np.save(os.path.join(model_path, "X.npy"), X)
# np.save(os.path.join(model_path, "Y.npy"), y)
# np.save(os.path.join(model_path, "X_syn.npy"), X_syn)
# np.save(os.path.join(model_path, "Y_syn.npy"), y_syn)