import numpy as np
import math
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse


def main(args):

	input_video = args.input_video
	in_folder = os.path.dirname(input_video)

	imag_file = os.path.join(in_folder, 'frame_buff_imag.txt')
	real_file = os.path.join(in_folder, 'frame_buff_real.txt')
	data_imag = np.loadtxt(imag_file)      # load imaginary part
	data_real = np.loadtxt(real_file)      # load real part
	number = min(len(data_imag), len(data_real))
	print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
	data_imag = data_imag[0:number]
	data_real = data_real[0:number]

	data_complex = data_real + 1j * data_imag                       # compute complex number
	fps = 120               # fps of UWB data
	desired_fps = 24        # fps in Vid2dop
	overlap = int(fps / desired_fps)             # overlap when do sliding window
	doppler_gt = []         # doppler data
	doppler_original = []
	for i in range(0, len(data_complex) - fps, overlap):
		d_fft = np.abs(np.fft.fft(data_complex[i:i+fps, :], axis=0))    # FFT
		d_fft = np.fft.fftshift(d_fft, axes=0)                          # shift
		doppler_original.append(d_fft)

		# sum over range
		d_fft = np.sum(d_fft, axis=1)
		# down sample from 120 bins to 32 bins, symmetrically
		indices = np.linspace(0, 59, 16).astype(int)
		indices = sorted(np.concatenate((indices, 119 - indices)))
		doppler_gt.append(d_fft[indices])
	doppler_gt = np.transpose(np.array(doppler_gt))
	doppler_original = np.array(doppler_original)
	# need to do normalization before visualization
	# sns.heatmap(doppler_gt)
	# plt.show()
	np.save(os.path.join(in_folder, 'doppler_original.npy'), doppler_original)
	np.save(os.path.join(in_folder, 'doppler_gt.npy'), doppler_gt)




if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file', default='../data/rgb.avi')

	args = parser.parse_args()

	main(args)



