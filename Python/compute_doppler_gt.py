import numpy as np
import math
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse
from helper import get_spectograms, color_scale
import cv2
import matplotlib


def main(args, input_video=""):
	DISCARD_BINS = [14, 15, 16]
	ARM_LEN = 30       # 150cm/5cm
	threshold = 0.1

	if input_video == "":
		input_video = args.input_video
	in_folder = os.path.dirname(input_video)
	vid_file_name = os.path.basename(input_video).split('.')[0]

	imag_file = os.path.join(in_folder, 'frame_buff_imag.txt')
	real_file = os.path.join(in_folder, 'frame_buff_real.txt')
	data_imag = np.loadtxt(imag_file)      # load imaginary part
	data_real = np.loadtxt(real_file)      # load real part
	number = min(len(data_imag), len(data_real))
	print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
	data_imag = data_imag[0:number]
	data_real = data_real[0:number]

	data_complex = data_real + 1j * data_imag                       # compute complex number
	# mm = np.mean(data_complex, axis=0)
	# data_complex = data_complex - mm
	fps = 120               # fps of UWB data
	desired_fps = 24        # fps in Vid2dop
	bin_num = 32
	overlap = int(fps / desired_fps)             # overlap when do sliding window
	doppler_gt = []         # doppler data
	doppler_original = []
	for i in range(0, len(data_complex) - bin_num, overlap):
		d_fft = np.abs(np.fft.fft(data_complex[i:i+bin_num, :], axis=0))    # FFT
		d_fft = np.fft.fftshift(d_fft, axes=0)                          # shift
		doppler_original.append(d_fft)

		# discard central rows
		fft_discard = np.copy(d_fft)
		for j in DISCARD_BINS:
			fft_discard[j, :] = np.zeros(188)
		# sns.heatmap(fft_discard)
		# plt.show()

		# select range bins with the highest energy
		mmax = np.max(fft_discard)
		row, column = np.where(fft_discard == mmax)
		row, column = row[0], column[0]
		left = column
		right = column + 1
		if column > 0:
			for left in range(column - 1, 0, -1):
				if np.max(fft_discard[:, left]) < threshold * mmax:
					break
		if column < 187:
			for right in range(column + 1, 188, 1):
				if np.max(fft_discard[:, right]) < threshold * mmax:
					break
		left = max(left, column - ARM_LEN)
		right = min(right, column + ARM_LEN)
		fft_gt = np.copy(d_fft[:, left:right])
		for j in DISCARD_BINS:      # normalize the central 2 rows
			mean_j = np.mean(fft_gt[j, :])
			fft_gt[j, :] -= mean_j
		# sns.heatmap(fft_gt)
		# plt.show()

		# sum over range
		fft_gt = np.sum(fft_gt, axis=1)
		# # down sample from 120 bins to 32 bins, symmetrically
		# indices = np.linspace(0, 58, 16).astype(int)
		# indices = sorted(np.concatenate((indices, 117 - indices)))
		# doppler_gt.append(d_fft[indices])
		doppler_gt.append(fft_gt)
	doppler_gt = np.array(doppler_gt)
	# doppler_gt = np.transpose(np.array(doppler_gt))
	doppler_original = np.array(doppler_original)
	np.save(os.path.join(in_folder, 'doppler_original.npy'), doppler_original)
	np.save(os.path.join(in_folder, 'doppler_gt.npy'), doppler_gt)

	model_path = args.model_path
	scale_vals = np.load(model_path + "scale_vals_new.npy")
	scale_vals[0] = max(scale_vals[0], np.max(doppler_gt))
	scale_vals[2] = min(scale_vals[2], np.min(doppler_gt))
	np.save(os.path.join(model_path, "scale_vals_new.npy"), scale_vals)

	return np.max(doppler_gt), np.min(doppler_gt)


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
def compute_doppler_gt_all(path='../data/2022_11_09/har1/', max_dopVal = 1.07, min_dopVal = 0):
	path = os.path.abspath(path)
	dires = os.listdir(path)
	for dire in dires:
		filename = os.path.join(path + '/' + dire, 'rgb.avi')
		mm, mi = main("", input_video=filename)
		max_dopVal = max(max_dopVal, mm)
		min_dopVal = min(min_dopVal, mi)
	print(max_dopVal)
	print(min_dopVal)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file',
	                    default='../data/2022_11_09/har1/2022_11_09_16_56_08_Jhair_sitting/rgb.avi')

	parser.add_argument('--model_path', type=str, help='Path to DL models')

	args = parser.parse_args()

	main(args)



