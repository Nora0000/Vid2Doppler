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
from helper import color_scale, chose_central_rangebins, first_peak
import matplotlib
from scipy.signal import find_peaks

path = "../data/2023_05_03/2023_05_03_16_34_39_mengjing_push_diagonal"

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
data_complex = (np.array(real[:num, :]) + 1j * np.array(imag[:num, :]))[1800:-1800, :]
range_profile = np.abs(data_complex)


doppler = []
doppler_bin_num = 32
DISCARD_BINS = [14, 15, 16, 17]

# locate torso
std_dis = np.std(range_profile, axis=0)[:94]
left, right = first_peak(std_dis)
print("left: {}, right: {}.".format(left, right))
np.savetxt(os.path.join(path, "std_of_range_profile.txt"), std_dis)

plt.axvline(x=left, color='r')
plt.axvline(x=right, color='r')
plt.plot(np.arange(0, std_dis.shape[0], 1), std_dis)
plt.title("first peak is {}~{}".format(left, right))
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
plt.ylabel("standard deviation")
plt.xlabel("range bin")
plt.show()

mean_dis = np.mean(range_profile, axis=0)
range_profile = range_profile - mean_dis
plt.axvline(x=left, color='r')
plt.axvline(x=right, color='r')
sns.heatmap(range_profile)
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

	fft_1d = np.sum(dop[:, left:right], axis=1)

	# select range bins with the highest energy
	# mmax = np.max(dop[:, :range_bin_num_])
	# row, column = np.where(dop == mmax)
	# row, column = row[0], column[0]
	#
	# fft_1d = np.zeros(doppler_bin_num)
	# fft_1d[row] = mmax

	# left = max(column-2, 0)
	# right = min(column+2, range_bin_num)
	# fft_central = np.copy(dop[:, left:right])
	# fft_1d = np.sum(fft_central, axis=1)

	doppler_1d.append(fft_1d)


doppler_1d = np.array(doppler_1d)
# np.save(os.path.join(path, 'doppler_gt.npy'), doppler_1d)
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
