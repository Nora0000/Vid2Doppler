import argparse
import os
import subprocess
import random
import math
import time
import numpy as np
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid
# from compute_doppler_gt import main
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from helper import color_scale, chose_central_rangebins
import matplotlib

path = "/home/mengjingliu/Vid2Doppler/data/2023_04_13/2023_04_13_18_34_33_Han_push_1"
imag = np.loadtxt(path + '/frame_buff_imag.txt')
real = np.loadtxt(path + '/frame_buff_real.txt')
num = min(len(imag), len(real))
data_complex = np.array(real[:num, :]) + 1j * np.array(imag[:num, :])
range_profile = np.abs(data_complex)
mean_dis = np.mean(range_profile, axis=0)
range_profile = range_profile - mean_dis
mean_comp = np.mean(data_complex, axis=0)
data_complex_ = data_complex - mean_comp
doppler = []
doppler_bin_num = 32
range_bin_num = 96
doppler_1d = []
for i in range(230, data_complex_.shape[0], 3):
	dop = np.abs(np.fft.fft(data_complex_[i - doppler_bin_num:i, :], axis=0))
	dop = np.fft.fftshift(dop, axes=0)  # shift
	dop = dop[:, :range_bin_num]
	dop[14:18, :] = np.zeros((4, range_bin_num))
	left, right = chose_central_rangebins(dop, range_bin_num)
	bg = max(np.max(dop[:, 0:left]), np.max(dop[:, right:]))
	dop[dop<bg] = 0
	fft_central = np.copy(dop[:, left:right])
	fft_1d = np.sum(fft_central, axis=1)
	# sns.heatmap(dop)
	# plt.show()
	# sns.heatmap(fft_central)
	# plt.show()
	doppler_1d.append(fft_1d)
	# sns.heatmap(fft_1d[:, np.newaxis])
	# plt.show()
	doppler.append(dop)

np.save(os.path.join(path, 'doppler_gt.npy'), np.array(doppler_1d))

doppler = np.array(doppler)
print(np.max(doppler))
print(np.min(doppler))
doppler[:, 14:17, :] = np.zeros((doppler.shape[0], 3, 96))
print(np.max(doppler))
print(np.min(doppler))
mmax = np.percentile(doppler, 99.99)
# mmax = np.max(doppler)
print(mmax)

height, width = doppler[0].shape[0], doppler[0].shape[1]
doppler[doppler>mmax] = mmax
flag = True
for d in doppler:
	dd = color_scale(d, matplotlib.colors.Normalize(vmin=np.min(doppler), vmax=mmax), "push/pull")
	if flag:
		out_vid = cv2.VideoWriter('han_push_1.mp4',
		                          cv2.VideoWriter_fourcc(*'mp4v'),
		                          24, (dd.shape[1], dd.shape[0]))
		flag = False
	out_vid.write(dd)
	# cv2.imshow("xx", dd)
out_vid.release()
