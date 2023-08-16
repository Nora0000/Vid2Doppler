import os
import numpy as np
import matplotlib
from helper import get_spectograms, root_mean_squared_error, color_scale, rolling_window_combine
from tensorflow.keras.models import load_model
import pickle
import cv2
import argparse
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def pass_zero(doppler_T, threshold=-0.4):
	dd = np.sum(doppler_T, axis=1)
	threshold = np.median(dd)
	return np.int64(dd>threshold)


def sliding_window(data, TIME_CHUNK=3, fps=24):
	re = []
	for i in range(len(data)-fps*TIME_CHUNK):
		re.append(data[i:i + TIME_CHUNK*fps])
	return np.array(re)


def peak_valley(doppler_T, flag):
	# if flag == 'gt':
	# 	threshold = 1.5
	# else:
	# 	threshold = 1
	threshold = 1.5
	re = []
	indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
	doppler_T = doppler_T[:, indices]
	mi = np.min(doppler_T)
	ma = np.max(doppler_T)
	doppler_T = (doppler_T - mi) / (ma - mi)
	pre = 0
	cnt = 0
	cur = 0
	for dop in doppler_T:
		left = np.sum(dop[0:14])
		right = np.sum(dop[14:])
		if left > threshold * right:
			re.append(-1)
		elif left * threshold < right:
			re.append(1)
		else:
			re.append(0)
		# if left > threshold * right:
		# 	if pre != 1 or (cnt >= 10 and cur == -1):
		# 		if cnt == 0:
		# 			re.append(-1)
		# 		else:
		# 			re.extend([-1] * cnt)
		# 		pre = -1
		# 		cur = 0
		# 		cnt = 0
		# 	elif cur == -1:
		# 		cnt += 1
		# 	else:
		# 		cur = -1
		# 		cnt = 1
		# elif left * threshold < right:
		# 	if pre != -1 or (cnt >= 10 and cur == 1):
		# 		re.extend([1] * cnt)
		# 		pre = 1
		# 		cur = 0
		# 		cnt = 0
		# 	elif cur == 1:
		# 		cnt += 1
		# 	else:
		# 		cur = 1
		# 		cnt = 1
		# else:
		# 	re.append(0)
		# 	pre = 0
		# 	cnt = 0
		# 	cur = 0
	a, b = butter(8, 0.8, 'lowpass')
	filtered_re = filtfilt(b, a, re)
	plt.plot(np.arange(0, len(re)), filtered_re)
	plt.show()
	return np.array(re)


def curve(doppler_T):
	threshold = 0.2
	peak = []
	valley = []
	for dop in doppler_T:
		indices = np.where(dop > threshold*np.max(dop))[0]
		left, right = indices[0], indices[-1]
		peak.append(right)
		valley.append(left)
	return peak, valley


def main(args, input_video="/home/mengjingliu/Vid2Doppler/data/2023_01_18/har3/2023_01_18_19_01_09_Han_movingbox/rgb.avi", model_path="../models/", if_doppler_gt=True):
	if model_path == "":
		model_path = args.model_path

	scale_vals = np.load(model_path + "scale_vals_new.npy")
	fps = 24
	TIME_CHUNK = 3
	max_dopVal = scale_vals[0]
	max_synth_dopVal = scale_vals[1]
	min_dopVal = scale_vals[2]
	min_synth_dopVal = scale_vals[3]
	mean_discard_bins = scale_vals[4:7]
	mean_synth_discard_bins = scale_vals[8:11]
	DISCARD_BINS = [14, 15, 16]

	if input_video == "":
		vid_f = args.input_video
	else:
		vid_f = input_video
	in_folder = os.path.dirname(vid_f)
	vid_file_name = vid_f.split("/")[-1].split(".")[0]

	# cap = cv2.VideoCapture(vid_f)

	# find common indices
	frames_common = np.load(os.path.join(in_folder, 'frames_common.npy')).astype(int)
	if os.path.isfile(os.path.join(in_folder, 'frames_new.npy')):
		frames = np.load(os.path.join(in_folder, 'frames_new.npy'), allow_pickle=True)
	else:
		frames = np.load(os.path.join(in_folder, 'frames.npy'), allow_pickle=True)
	indices = np.isin(frames, frames_common)

	synth_doppler_dat_f = in_folder + "/output/" + vid_file_name + "/synth_doppler.npy"
	synth_doppler_dat = np.load(synth_doppler_dat_f)[indices, :]
	synth_doppler_dat[:, DISCARD_BINS] -= mean_synth_discard_bins
	# synth_spec_pred = get_spectograms(synth_doppler_dat, TIME_CHUNK, fps, synthetic=True, zero_pad=True)
	synth_spec_pred = synth_doppler_dat.astype("float32")
	# normalization
	synth_spec_test = (synth_spec_pred - min_synth_dopVal) / (max_synth_dopVal - min_synth_dopVal)
	# peak, valley = curve(synth_spec_test)
	# plt.plot(np.array(peak))
	# plt.plot(np.array(valley))
	seq = pass_zero(synth_spec_test)
	# synth_wave = peak_valley(synth_spec_test, 'synth')
	# synth_wave_spec = sliding_window(synth_wave)
	# synth_seg = np.array([synth_wave[i:i+24*9] for i in range(0, len(synth_wave)-24*9, 24*9)])



	dop_spec_test = np.zeros_like(synth_spec_test)
	# doppler_dat_pos_f = in_folder + "/../" + "/doppler_gt.npy"
	doppler_dat_pos_f = in_folder + "/doppler_gt.npy"
	doppler_dat_pos = np.load(doppler_dat_pos_f)
	doppler_dat_pos[:, DISCARD_BINS] -= mean_discard_bins
	# dop_spec = get_spectograms(doppler_dat_pos, TIME_CHUNK, fps, zero_pad=True)
	dop_spec = doppler_dat_pos.astype("float32")
	dop_spec_test = (dop_spec - min_dopVal) / (max_dopVal - min_dopVal)
	seq_gt = pass_zero(dop_spec_test, threshold=3)
	# gt_wave = peak_valley(dop_spec_test, 'gt')
	# gt_wave_spec = sliding_window(gt_wave)
	# gt_seg = np.array([gt_wave[i:i+24*9] for i in range(0, len(gt_wave)-24*9, 24*9)])

	np.savetxt(os.path.join(in_folder, "sequence_synth.txt"), seq, delimiter=',')
	np.savetxt(os.path.join(in_folder, "sequence_gt.txt"), seq_gt, delimiter=',')

	return 0


def plot_all():
	path = '../data/2022_11_09/'
	path_list = os.listdir(path)
	i = 0
	for path_i in path_list:
		path_i = os.path.join(os.path.abspath(path), path_i)
		for file in os.listdir(path_i):
			input_video = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute.mp4')
			main("", input_video=input_video, model_path='../models/', if_doppler_gt=True)
			print(i)
			i += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file',
	                    default='../data/2022_11_09/har1/2022_11_09_16_56_08_Jhair_sitting/rgb.avi')

	parser.add_argument('--model_path', type=str, help='Path to DL models', default='../models/')

	parser.add_argument('--doppler_gt', help='Doppler Ground Truth is available for reference', action='store_true',
	                    default=True)

	args = parser.parse_args()

	main(args)
