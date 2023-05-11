import numpy as np
import cv2
import sys
import imutils
from matplotlib import cm
from tensorflow.python.keras import backend as K
import tensorflow as tf
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt


def first_peak(std_dis):
	peaks, _ = find_peaks(std_dis, height=np.percentile(std_dis, 50))
	peaks_, _ = find_peaks(-std_dis)

	first_peak = peaks[0]
	second_peak = peaks[1]
	left = 0
	right = second_peak
	for pp in peaks_:
		if pp < first_peak:
			left = max(left, pp)
		else:
			break

	print("first peak: {}, left: {}, right: {}".format(first_peak, left, right))

	return left, right


def downsample_bins(original=120, target=32):
	# down sample from 120 bins to 32 bins, symmetrically
	indices = np.linspace(0, int(original/2-1), int(target/2)).astype(int)
	indices = sorted(np.concatenate((indices, original-3 - indices)))
	return indices


def chose_central_rangebins(d_fft, range_bin_num=188, torso_loc=[], DISCARD_BINS=[14, 15, 16, 17], threshold=0.1, ARM_LEN=30, doppler_bin=32):
	fft_discard = np.copy(d_fft)
	# discard velocities around 0
	for j in DISCARD_BINS:
		fft_discard[j, :] = np.zeros(range_bin_num)
	# select range bins with the highest energy
	mmax = np.max(fft_discard)
	row, column = np.where(fft_discard == mmax)
	row, column = row[0], column[0]
	# if column in torso_loc:
	# 	fft_discard_column = np.copy(fft_discard)
	# 	fft_discard_column[:, torso_loc] = np.zeros((doppler_bin, len(torso_loc)))
	# 	mmax = np.max(fft_discard_column)
	# 	row, column = np.where(fft_discard_column == mmax)
	# 	row, column = row[0], column[0]
	# 	if column >= 2:
	# 		return column - 2, column + 3
	# 	else:
	# 		return 0, column+3
	if column >= 2:
		return column - 2, column + 3
	else:
		return 0, column + 3
	left = column
	right = column + 1
	if column > 0:
		for left in range(column - 1, 0, -1):
			if np.max(fft_discard[:, left]) < threshold * mmax:
				break
	if column < range_bin_num-1:
		for right in range(column + 1, range_bin_num, 1):
			if np.max(fft_discard[:, right]) < threshold * mmax:
				break
	left = max(left, column - ARM_LEN)
	right = min(right, column + ARM_LEN)
	return left, right


def rolling_window_combine(X_in):
	for i in range(1,len(X_in)):
		X_in[i][:,:-1] = X_in[i-1][:,1:]
	return X_in

def color_scale(img, norm,text=None):
	if len(img.shape) == 2:
		img = cm.magma(norm(img),bytes=True)
	img = imutils.resize(img, height=300)
	if img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
	if text is not None:
		img = cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
	return img

def rolling_average(X_in):
	for i in range(1,X_in.shape[0]-1):
		X_in[i] = (X_in[i] + X_in[i+1] + X_in[i-1])/3
	return X_in

def root_mean_squared_error(y_true, y_pred):
	indices = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
	return K.sqrt(K.mean(K.square((tf.gather(y_pred, indices, axis=1)*255) - (tf.gather(y_true, indices, axis=1)*255))))

def get_spectograms(dop_dat, t_chunk, frames_per_sec, bin_num=32, t_chunk_overlap=None, synthetic=False,zero_pad=False):
	frame_overlap = 1
	if t_chunk_overlap is not None:
		frame_overlap = int(t_chunk_overlap * frames_per_sec)
	frame_chunk = int(t_chunk * frames_per_sec)
	if zero_pad == True:
		zero_padding = np.zeros((bin_num,frame_chunk-1))
		dop_dat_spec = np.hstack((zero_padding,np.transpose(dop_dat)))
	else:
		dop_dat_spec = np.transpose(dop_dat)
	spectogram = []
	if zero_pad == True:
		for i in range(0,len(dop_dat), frame_overlap):
			spec = dop_dat_spec[:,i:i+frame_chunk]
			if synthetic == True:
					spec = cv2.GaussianBlur(spec,(5,5),0)
			spectogram.append(spec)
	else:
		for i in range(0,len(dop_dat)-frame_chunk, frame_overlap):
			spec = dop_dat_spec[:,i:i+frame_chunk]
			if synthetic == True:
				if zero_pad == True:
					spec = cv2.GaussianBlur(spec,(5,5),0)
			spectogram.append(spec)
	spectogram = np.array(spectogram)
	return spectogram


def compute_fr_from_ts(path = "/home/mengjingliu/Vid2Doppler/data/2023_04_24/2023_04_24_17_44_27_ADL_zh"):

	rgb_ts = np.loadtxt(os.path.join(path, "rgb_ts.txt"))

	pre_ts = rgb_ts[0]
	frame_rates = []
	cnt = 0
	for ts in rgb_ts:
		if ts - pre_ts < 1:
			cnt += 1
		else:
			frame_rates.append(cnt)
			cnt = 0
			pre_ts = ts
	frame_rates = np.array(frame_rates)
	plt.plot(np.arange(0, len(frame_rates), 1), frame_rates)
	plt.show()
	print(np.mean(frame_rates))