import os
# import subprocess
# import random
# import math
# import time
import numpy as np
# import shutil
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
import cv2
from helper import color_scale, chose_central_rangebins
import matplotlib

path = "../data/2023_05_02/2023-05-03-17-16-46_mengjing_zigzag_diagonal_fps180"

# synDop = np.load(os.path.join(path, "output/rgb/synth_doppler.npy"))
# sns.heatmap(synDop)
# plt.show()
#
# gtDop = np.load(os.path.join(path, "doppler_gt.npy"))
# sns.heatmap(gtDop)
# plt.show()


imag = np.loadtxt(path + '/frame_buff_imag.txt')
real = np.loadtxt(path + '/frame_buff_real.txt')
num = min(len(imag), len(real))
data_complex = (np.array(real[:num, :]) + 1j * np.array(imag[:num, :]))[900:-900, :]
range_profile = np.abs(data_complex)
mean_dis = np.mean(range_profile, axis=0)
range_profile = range_profile - mean_dis
mean_dis_ = np.mean(range_profile, axis=0)
sns.heatmap(range_profile)
plt.show()

doppler = []
doppler_bin_num = 32
DISCARD_BINS = [14, 15, 16, 17]

# locate torso
std_dis = np.std(range_profile, axis=0)[:94]
plt.plot(np.arange(0, std_dis.shape[0], 1), std_dis)
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
plt.ylabel("standard deviation")
plt.xlabel("range bin")
plt.show()

range_bin_num = 94

data_complex_ = data_complex[:, :range_bin_num]

range_bin_num_ = 12
doppler_1d = []
for i in range(doppler_bin_num, data_complex_.shape[0], 2):
	dop = np.abs(np.fft.fft(data_complex_[i - doppler_bin_num:i, :], axis=0))
	dop = np.fft.fftshift(dop, axes=0)  # shift
	doppler.append(dop)
	dop = dop[:, :range_bin_num]
	dop[14:18, :] = np.zeros((4, range_bin_num))

	# select range bins with the highest energy
	mmax = np.max(dop[:, :range_bin_num_])
	row, column = np.where(dop == mmax)
	row, column = row[0], column[0]

	fft_1d = np.zeros(doppler_bin_num)
	fft_1d[row] = mmax

	# left = max(column-2, 0)
	# right = min(column+2, range_bin_num)
	# fft_central = np.copy(dop[:, left:right])
	# fft_1d = np.sum(fft_central, axis=1)

	doppler_1d.append(fft_1d)


doppler_1d = np.array(doppler_1d)
np.save(os.path.join(path, 'doppler_gt.npy'), doppler_1d)
sns.heatmap(np.array(doppler_1d))
plt.show()

sns.heatmap(doppler_1d[2000:3000, :])
plt.show()


doppler = np.array(doppler)
np.save(os.path.join(path, 'range_doppler_rb94.npy'), doppler)
# doppler_mean = np.mean(doppler, axis=0)
# doppler = (doppler - doppler_mean)[:, :, :range_bin_num]
print(np.max(doppler))
print(np.min(doppler))
doppler[:, 14:17, :] = np.zeros((doppler.shape[0], 3, range_bin_num))
# mmax = np.percentile(doppler, 99.99)
mmax = np.max(doppler)
mmin = np.min(doppler)

height, width = doppler[0].shape[0], doppler[0].shape[1]
# doppler[doppler>mmax] = mmax
flag = True
for d in doppler:

	# mmax = np.max(d)
	# row, column = np.where(d == mmax)
	# row, column = row[0], column[0]
	# dd = np.zeros((doppler_bin_num, range_bin_num))
	# # dd[row, column] = d[row, column]
	# dd[row, column] = 1

	dd = color_scale(d, matplotlib.colors.Normalize(vmin=mmin, vmax=mmax), "push")
	if flag:
		out_vid = cv2.VideoWriter(os.path.join(path, 'mengjing_push_new.mp4'),
		                          cv2.VideoWriter_fourcc(*'mp4v'),
		                          24, (dd.shape[1], dd.shape[0]))
		flag = False
	out_vid.write(dd)
	# cv2.imshow("xx", dd)
out_vid.release()
