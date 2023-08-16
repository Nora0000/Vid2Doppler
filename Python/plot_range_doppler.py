import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helper import color_scale, first_peak
import matplotlib
import cv2


path = "/home/mengjingliu/Vid2Doppler/data/2023_05_04/2023_05_04_16_58_18_mengjing_squat"

imag = np.loadtxt(path + '/frame_buff_imag.txt')
real = np.loadtxt(path + '/frame_buff_real.txt')
num = min(len(imag), len(real))
data_complex = (np.array(real[:num, :]) + 1j * np.array(imag[:num, :]))[1800:-1800, :]
range_profile = np.abs(data_complex)
doppler_bin_num = 32
DISCARD_BINS = [14, 15, 16, 17]

std_dis = np.std(range_profile, axis=0)[:94]
left, right = first_peak(std_dis)
print("left: {}, right: {}.".format(left, right))

doppler = np.load(os.path.join(path, "doppler_original.npy"))
# range_bin_num = 94

doppler[:, 14:18, :] = np.zeros((doppler.shape[0], 4, doppler.shape[2]))
# mmax = np.percentile(doppler, 99.99)
mmax = np.max(doppler)
mmin = np.min(doppler)

height, width = doppler[0].shape[0], doppler[0].shape[1]
flag = True
for d in doppler:

	dd = color_scale(d[:, left:right], matplotlib.colors.Normalize(vmin=mmin, vmax=mmax), " ")
	if flag:
		out_vid = cv2.VideoWriter(os.path.join(path, 'range_doppler_cut.mp4'),
		                          cv2.VideoWriter_fourcc(*'mp4v'),
		                          24, (dd.shape[1], dd.shape[0]))
		flag = False
	out_vid.write(dd)
	# cv2.imshow("xx", dd)
out_vid.release()
