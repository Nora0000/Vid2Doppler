import torch
import os
import numpy as np
from torch import nn
import seaborn as sns
from compute_doppler_gt_new import align_index
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d


def find_segment_indices(seg_dt_start, seg_dt_end, times_dt):
	"""
	traverse timestamps of each data frame to find the segment fitting in seg_dt_start and seg_dt_end. return the indices.

	Args:
		seg_dt_start: datetime
		seg_dt_end:
		times_dt:
		augment:

	Returns: indices of frames which is collected between datetime seg_dt_start, seg_dt_end

	"""
	i = 0
	# data_indices = []
	while i < len(times_dt) and times_dt[i] < seg_dt_start:
		i += 1
	start = i
	while i < len(times_dt) and times_dt[i] <= seg_dt_end:
		# data_indices.append(i)
		i += 1
	stop = i
	return start, stop


def segment_align(path, path_of_segmentation, filename_segmentation="segmentation.npy", fps=180):
	"""
	segment data of new viewpoint based on the segmentation from segmented viewpoint.
	Need to add time lag between different computers.

	path: data path of new viewpoint
	path_of_segmentation: path of segmented view point
	Returns:
		object:
	"""
	segs = np.load(os.path.join(path_of_segmentation, filename_segmentation))
	uwb_indices = np.load(os.path.join(path_of_segmentation, "uwb_indices.npy"))
	times = np.loadtxt(os.path.join(path_of_segmentation, "times.txt"))
	start = times[0]
	
	times_new = np.loadtxt(os.path.join(path, "times.txt"))
	uwb_indices_new = np.load(os.path.join(path, "uwb_indices.npy"))
	start_new = times_new[0]
	
	segs_new = []
	for seg in segs:
		uwb1, uwb2 = uwb_indices[seg[0]], uwb_indices[seg[1]]
		# t1, t2 = times[uwb1], times[uwb2]
		# start_ind, stop_ind = find_segment_indices(t1+1, t2+1.5, times_new)
		# times.txt不准确。用开始时间和时间间隔、帧率计算更准确
		t1, t2 = start + uwb1/fps, start + uwb2/fps
		start_ind, stop_ind = align_index(t1-0.9, start_new, fps), align_index(t2-0.6, start_new, fps)
		start_ind, stop_ind = find_segment_indices(start_ind, stop_ind, uwb_indices_new)
		segs_new.append([start_ind, stop_ind])
	
	np.save(os.path.join(path, filename_segmentation), np.array(segs_new))


def interpolate(d, height=32, width=50):    # this can do up sample and down sample
	return nn.UpsamplingBilinear2d(size=(height, width))(torch.tensor(d[np.newaxis, np.newaxis, :, :])).numpy()[0, 0, :, :]


def load_data(path, filename, path_seg, filter=False, filename_seg="segmentation.npy", label=0, path_of_segmentation_rely=None, seg_length=50):
	# if not os.path.exists(os.path.join(path_seg, filename_seg)):
	# 	print("no segmentation file. path: {}".format(path))
	# 	segment_align(path_seg, path_of_segmentation_rely)
	
	segs = np.load(os.path.join(path_seg, filename_seg))
	doppler = np.transpose(np.load(os.path.join(path, filename)))
	if filename.split('_')[0] == "synth":
		# align between synthetic doppler and real doppler, so they use the same segmentation result
		frames = np.load(os.path.join(path_seg, "frames.npy"))
		frames_common = np.load(os.path.join(path_seg, "frames_common.npy"))
		indices = np.isin(frames, frames_common)
		doppler = doppler[:, indices]
		if filter:
			for i in range(doppler.shape[1]):
				doppler[:, i] = gaussian_filter1d(doppler[:, i], 3)
				
	
	d_seg = []
	for seg in segs:
		if seg[1] - seg[0] <= 0:
			continue
		d = doppler[:, seg[0]:seg[1]]
		# sns.heatmap(d)
		# plt.show()
		d_seg.append(interpolate(d, width=seg_length))		# align dimension of all segments
	return np.array(d_seg), np.ones(len(segs))*int(label)


def augment_data(X, Y):
	"""
	augment data set
	Args:
		X: doppler data
		Y: label
	Returns: augmented data set, including original data set

	"""
	
	bin_num, seg_length = X.shape[1], X.shape[2]
	
	X_aug = []
	Y_aug = []
	for x, y in zip(X, Y):
		# uniform filter
		X_aug.append(x)
		Y_aug.append(y)
		# sizes = [2, 3]
		# for s in sizes:
		# 	d_ave = np.copy(x)
		# 	d_ave = uniform_filter(d_ave, size=s, mode="nearest")
		# 	X_aug.append(d_ave)
		# 	Y_aug.append(y)
		
		shifts = [1, 2, 3, 4]
		# shifts = np.arange(1, 10)
		for shift in shifts:
			# left
			d_shift = np.copy(x)
			d_shift[:, :seg_length - shift] = d_shift[:, shift:]
			d_shift[:, -shift:] = np.zeros((bin_num, shift))
			X_aug.append(d_shift)
			Y_aug.append(y)
			
			# right
			d_shift = np.copy(x)
			d_shift[:, shift:] = d_shift[:, 0:seg_length - shift]
			d_shift[:, :shift] = np.zeros((bin_num, shift))
			X_aug.append(d_shift)
			Y_aug.append(y)
			
		# peper salt noise
		# SNR = 0.5
		# d_pepper = np.copy(d_interp)
		# mmax = np.max(d_pepper)
		# mmin = np.min(d_pepper)
		# mask = np.random.choice((0, 1, 2), size=(32, seg_length), p=[SNR, (1-SNR)/2, (1-SNR)/2])
		# d_pepper[mask==1] = mmax
		# d_pepper[mask==2] = mmin
		# sns.heatmap(d_pepper)
		# plt.show()
		
		
		# shift: capture window to left by 1
		
		# Gaussian blur
		# sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
		# # sigmas = np.arange(1, 20) / 40
		# for sigma in sigmas:
		# 	d_gau = np.copy(x)
		# 	d_gau = gaussian_filter(d_gau, sigma=sigma)
		# 	X_aug.append(d_gau)
		# 	Y_aug.append(y)
		
	return np.array(X_aug), np.array(Y_aug)


if __name__ == "__main__":
	"""
	label:
	circle: 1, push: 2, step: 4, sit: 3, stand: 5, bend: 6
	
	"""
	# load doppler (real or synthetic) and segmentation, aggregate dataset of different activities and generate labels.
	act = ["push", "step", "sit", "circle", "bend"]
	path = "/home/mengjingliu/Vid2Doppler/data/2023_11_17/HAR3"
	filename = "doppler_gt.npy"
	# filename = "synth_doppler.npy"
	act = os.listdir(path)
	X = []
	Y = []
	for ac in act:
		# path_data = os.path.join(path, ac) + "/output/rgb/"
		path_data = os.path.join(path, ac)
		a = ac.split('_')[-1]
		# if a == "bend":
		# 	# segment_align(path_data, "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/2023_07_19_22_07_44_bend", fps=180)
		# 	data, _ = load_data(path_data, filename, os.path.join(path, ac))
		# 	label = np.array([2] * len(data))
		# 	np.save(os.path.join(path, "X_bend.npy"), data)
		# 	np.save(os.path.join(path, "Y_bend.npy"), label)
		# 	exit(0)
		if a == "sit":
			# segment_align(path_data,
			# 			  "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/2023_07_19_21_59_01_sit",
			# 			  "segmentation_sit.npy", fps=180)
			data, _ = load_data(path_data, filename, os.path.join(path, ac), filename_seg="segmentation_sit.npy")
			# data = augment_data(path_data,
			#                 filename, os.path.join(path, ac), filename_seg="segmentation_sit.npy")
			label = np.array([3] * len(data))
			np.save(os.path.join(path, "X_sit.npy"), data)
			np.save(os.path.join(path, "Y_sit.npy"), label)
			#
			# segment_align(path_data,
			# 			  "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/2023_07_19_21_59_01_sit",
			# 			  "segmentation_stand.npy", fps=180)
			data, _ = load_data(path_data, filename, os.path.join(path, ac), filename_seg="segmentation_stand.npy")
			label = np.array([5] * len(data))
			np.save(os.path.join(path, "X_stand.npy"), data)
			np.save(os.path.join(path, "Y_stand.npy"), label)
		else:
			continue
	