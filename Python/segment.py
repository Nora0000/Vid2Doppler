import numpy as np
import sys
from matplotlib import cm
import os
import matplotlib.pyplot as plt
from helper import apply_highpass_filter, apply_lowpass_filter
from math import log2
import seaborn as sns


def segment(path, filename, title="segment", threshold=2, width=10, doppler_bin=32, DISCARD_BINS = [15, 16, 17], if_save=False):
	"""
compute segmentation of doppler_time data. deternimne if a frame is "active" or "static".
With 10 continuous active frames, an activity starts.
With 10 continuous static frames, an activity ends.
	Args:
	 title:
		path:
		filename:
		threshold: threshold to determine "static" or "active" frame
		width: width of the window
		doppler_bin:

	Returns:

	"""
	d = np.load(os.path.join(path, filename))[120:-120, :]
	d[:, DISCARD_BINS] = np.zeros((len(d), len(DISCARD_BINS)))
	# d = apply_lowpass_filter(d, 0.3)
	
	static = []
	active = []
	status = "static"
	a_cnt = 0
	s_cnt = 0
	sigmas = []
	for i in range(len(d)):
		Epeak = np.max(d[i, :])
		Epeak_ = np.argmax(d[i, :])
		s = 0
		if Epeak_ >= 2:
			s += np.sum(d[i, 0:Epeak_-2])
		if Epeak_ <= doppler_bin-2:
			s += np.sum(d[i, Epeak_+2:])
		Enoise = s / (doppler_bin-4)
		sigma = log2((Epeak + Enoise)/Enoise)
		sigmas.append(sigma)
		if sigma > threshold:
			a_cnt += 1
			s_cnt = 0
		else:
			s_cnt += 1
			a_cnt = 0
		if status == "static" and a_cnt >= width:   # 9 frames delay
			status = "active"
		if status == "active" and s_cnt >= width:   # 9 frames delay
			status = "static"
		if status == "active":
			active.append(i)
		else:
			static.append(i)
	
	# dh = apply_highpass_filter(d, 0.3)
	# sns.heatmap(np.transpose(dh))
	# plt.show()
	
	sns.heatmap(np.transpose(d))
	plt.plot(sigmas, label="{}".format(chr(951)))
	plt.scatter(np.array(static)-(width-1), np.ones(len(static))*8, label="static")
	plt.scatter(np.array(active)-(width-1), np.ones(len(active))*10, label="active")
	plt.xlabel("time (second)")
	plt.ylabel("doppler FFT (Hz)")
	# plt.xticks([0, 240, 480, 720, 960], ['0', '3', '6', '9', '12'])
	# plt.xticks([0, 240, 480], ['0', '3', '6'])
	plt.yticks([0, 2, 16, 31], ['0', '2', '90', '180'])
	plt.title(title)
	plt.legend()
	plt.savefig(os.path.join(path, "segment_sample.png"))
	plt.show()
	
	# plt.plot(sigmas)
	# plt.grid()
	# plt.show()
	
	segs = []
	active = (np.array(active) - (width-1))     # subtract the (width-1) frames delay
	seg = [active[0]]
	for i in range(1, len(active)):
		if active[i] - active[i-1] == 1:
			seg.append(active[i])
		else:
			segs.append(seg)
			seg = [active[i]]
	segs.append(seg)
	
	if if_save:
		np.save(os.path.join(path, "segmentation_sample.npy"), np.array([[seg[0]+120, seg[-1]+120] for seg in segs]))
	
	ll = np.array([len(seg) for seg in segs])
	ll = np.sort(ll)
	ll_c = np.bincount(ll)
	plt.plot(ll_c)
	plt.grid()
	plt.title("segment length distribution")
	plt.savefig(os.path.join(path, "segment_length_distribution.png"))
	plt.show()
	
	return segs, sigmas
	

if __name__ == "__main__":
	path = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6/2023_07_19_21_26_50_step"
	filename = "doppler_gt.npy"
	# segs, sigmas = segment(path, filename, width=10, threshold=2.6, DISCARD_BINS=[15, 16, 17])     # sit4
	# segs, sigmas = segment(path, filename, width=10, threshold=2, DISCARD_BINS=[15, 16, 17])     # push4, circle6
	# segs, sigmas = segment(path, filename, width=3, threshold=2, DISCARD_BINS=[15, 16, 17])     # step4
	segs, sigmas = segment(path, filename, width=3, threshold=2.3, DISCARD_BINS=[15, 16, 17])     # step6
