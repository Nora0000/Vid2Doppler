import numpy as np
import math
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse
from helper import get_spectograms, color_scale, chose_central_rangebins
import cv2
import matplotlib


def main(args, input_video="", model_path=""):
	# DISCARD_BINS = [14, 15, 16, 17]
	ARM_LEN = 30       # 150cm/5cm
	threshold = 0.1
	bin_num = 16
	DISCARD_BINS = [6, 7, 8]
	range_bin_num = 96

	if model_path == "":
		model_path = args.model_path
	scale_vals = np.load(model_path + "scale_vals_new.npy")

	if input_video == "":
		input_video = args.input_video
	in_folder = os.path.dirname(input_video)
	vid_file_name = os.path.basename(input_video).split('.')[0]

	# imag_file = os.path.join(in_folder, '../frame_buff_imag.txt')
	# real_file = os.path.join(in_folder, '../frame_buff_real.txt')
	# ts_uwb_file = os.path.join(in_folder, '../times.txt')
	# ts_rgb_file = os.path.join(in_folder, '../rgb_ts.txt')
	imag_file = os.path.join(in_folder, 'frame_buff_imag.txt')
	real_file = os.path.join(in_folder, 'frame_buff_real.txt')
	ts_uwb_file = os.path.join(in_folder, 'times.txt')
	ts_rgb_file = os.path.join(in_folder, 'rgb_ts.txt')
	ts_uwb = np.loadtxt(ts_uwb_file)  # load timestamps
	ts_rgb = np.loadtxt(ts_rgb_file)

	data_imag = np.loadtxt(imag_file)      # load imaginary part
	data_real = np.loadtxt(real_file)      # load real part
	number = min(len(data_imag), len(data_real))
	print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
	data_imag = data_imag[0:number]
	data_real = data_real[0:number]

	data_complex = data_real + 1j * data_imag  # compute complex number

	range_profile = np.abs(data_complex)
	std_dis = np.std(range_profile[800:, :range_bin_num], axis=0)
	indices = np.argsort(std_dis)
	torso_loc = indices[-6:]

	range_bin_num = np.min(torso_loc)

	mean_comp = np.mean(data_complex, axis=0)
	data_complex = data_complex - mean_comp

	if os.path.isfile(in_folder + "/frames_new.npy"):
		frames = np.load(in_folder + "/frames_new.npy", allow_pickle=True)
	else:
		frames = np.load(in_folder + "/frames.npy", allow_pickle=True)

	doppler_gt = []  # doppler data
	doppler_original = []

	frames_common = []      # frame indices of video which have corresponding doppler ground truth
	uwb_indices = []        # corresponding indices of video frames in uwb data
	uwb_idx = bin_num
	for frame_idx in frames[:-1]:
		rgb_ts = float(ts_rgb[frame_idx])

		if rgb_ts - float(ts_rgb[frame_idx-11]) > 0.6:      # drop frames which are not continuous with previous ones in time
			print(" rgb frame not continuous. frame index: {}".format(frame_idx))
			continue

		if rgb_ts < float(ts_uwb[uwb_idx-1]):
			continue
		while rgb_ts > float(ts_uwb[uwb_idx]):
			uwb_idx += 1
			if uwb_idx >= len(ts_uwb):
				break
		if uwb_idx >= len(ts_uwb):
			break
		if float(ts_uwb[uwb_idx - 1]) <= rgb_ts <= float(ts_uwb[uwb_idx]):
			frames_common.append(frame_idx)
			uwb_indices.append(uwb_idx)
			d_fft = np.abs(np.fft.fft(data_complex[uwb_idx-bin_num:uwb_idx, :], axis=0))  # FFT
			d_fft = np.fft.fftshift(d_fft, axes=0)  # shift
			d_fft[DISCARD_BINS, :] = np.zeros((len(DISCARD_BINS), d_fft.shape[1]))
			doppler_original.append(d_fft)

			left, right = chose_central_rangebins(d_fft[:, :range_bin_num], range_bin_num, torso_loc=torso_loc, DISCARD_BINS=DISCARD_BINS)
			if left > 0 and right < range_bin_num:
				bg = max(np.max(d_fft[:, 0:left]), np.max(d_fft[:, right:]))
			elif left == 0:
				bg = np.max(d_fft[:, right:])
			elif right == range_bin_num:
				bg = np.max(d_fft[:, 0:left])
			else:
				bg = 0

			fft_gt = np.copy(d_fft[:, left:right])
			fft_gt[fft_gt < bg] = 0

			fft_gt[DISCARD_BINS, :] = np.zeros((len(DISCARD_BINS), right-left))

			scale_vals[0] = max(scale_vals[0], np.max(fft_gt))
			scale_vals[2] = min(scale_vals[2], np.min(fft_gt))

			# sum over range
			fft_gt = np.sum(fft_gt, axis=1)
			doppler_gt.append(fft_gt)
	doppler_gt = np.array(doppler_gt)
	doppler_original = np.array(doppler_original)
	np.save(os.path.join(in_folder, 'doppler_original.npy'), doppler_original)
	np.save(os.path.join(in_folder, 'doppler_gt.npy'), doppler_gt)
	np.save(os.path.join(in_folder, 'frames_common.npy'), np.array(frames_common))
	np.save(os.path.join(in_folder, 'uwb_indices.npy'), np.array(uwb_indices))

	number = len(doppler_gt)
	# scale_vals[4:7] = (scale_vals[4:7] * scale_vals[7] + np.mean(doppler_gt[:, DISCARD_BINS], axis=0) * number) / (
	# 			scale_vals[7] + number)
	# scale_vals[7] += number
	# np.save(os.path.join(model_path, "scale_vals_new.npy"), scale_vals)

	return scale_vals


def plot_doppler_gt(doppler_gt, in_folder, vid_file_name, fps=24):
	# plot doppler ground truth
	dop_spec = get_spectograms(doppler_gt, 3, 24, zero_pad=True)
	dop_spec = dop_spec.astype("float32")

	max_dopVal = np.max(dop_spec)
	min_dopVal = np.min(dop_spec)

	dop_spec_test = (dop_spec - min_dopVal) / (max_dopVal - min_dopVal)


	for idx in range(0, 1000):
		original_dop = color_scale(dop_spec_test[idx], matplotlib.colors.Normalize(vmin=0, vmax=np.max(dop_spec_test)),
		                           "Real World Doppler")
		if idx == 0:
			print(original_dop.shape)
			height, width = original_dop.shape[0], original_dop.shape[1]
			out_vid = cv2.VideoWriter(in_folder + '/' + vid_file_name + '_output_signal.mp4',
			                          cv2.VideoWriter_fourcc(*'mp4v'),
			                          fps, (width, height))
		out_vid.write(original_dop)

	out_vid.release()


# compute doppler_gt for all data from one sensor, return max value and min value of doppler
def compute_doppler_gt_all(path='../data/2022_11_09/'):
	path_list = os.listdir(path)
	i = 0
	scales = np.zeros(8)
	for path_i in path_list:
		path_i = os.path.join(os.path.abspath(path), path_i)
		for file in os.listdir(path_i):
			input_video = os.path.join(path_i + '/' + file, 'rgb.avi')
			scales = main("", input_video=input_video, model_path='../models/')
			print(i)
			i += 1
	print(scales)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file',
	                    default='/home/mengjingliu/Vid2Doppler/data/2023_04_24/2023_04_24_17_34_30_ADL_zx/rgb.avi')

	parser.add_argument('--model_path', type=str, help='Path to DL models', default="../models/")

	args = parser.parse_args()

	main(args)



