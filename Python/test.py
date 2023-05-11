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
from helper import color_scale, chose_central_rangebins, compute_fr_from_ts
import matplotlib
from scipy.signal import find_peaks

# compute_fr_from_ts("/home/mengjingliu/Vid2Doppler/data/2023_05_04/2023_05_04_18_07_20_mengjing_push")



scale_vel = np.loadtxt(os.path.join("../models/", "scale_velocity.txt"))
# v_min = scale_vel[0]
# v_max = scale_vel[1]
v_min = 1
v_max = -1

path = "/home/mengjingliu/Vid2Doppler/data/2023_05_05/2023_05_04_18_07_20_mengjing_push"

if os.path.isfile(path + \
                  "/frames_new.npy"):
	frames = np.load(path + \
	                 "/frames_new.npy", allow_pickle=True)
else:
	frames = np.load(path + \
	                 "/frames.npy", allow_pickle=True)
print("frames: ", len(frames))

cnt = 0
velocity_list = []
max_lim = []
for frame_idx in frames:
	if frame_idx < 300 or frame_idx > frames[-1]-300:
		continue
	gen_doppler = np.genfromtxt(path + "/output/rgb/frame_velocity/frame_%06d.csv" % frame_idx, delimiter=',')
	velocity = gen_doppler[gen_doppler[:, 1]==1, 0]
	# hist = np.histogram(velocity, bins=np.linspace(-3, 3, num=32 + 1))[0]  # fps=180
	# for bin_idx in [14, 15, 16]:  # don't ignore stationary parts.
	# 	hist[bin_idx] = 0

	velocity_list.extend(velocity)

	max_lim.append(np.max(np.abs(velocity)))


	cnt += 1
	if cnt <= 500:
		# sort data
		velocity_sorted = np.sort(velocity)

		# calculate CDF values
		y = 1. * np.arange(len(velocity_sorted)) / (len(velocity_sorted) - 1)

		# plot CDF
		plt.plot(velocity_sorted, y)

	# v_max = max(v_max, np.max(velocity))
	# v_min = min(v_min, np.min(velocity))
	# synth_doppler_dat.append(hist / gen_doppler.shape[0])

plt.show()


# sort data
velocity_sorted = np.sort(np.array(max_lim))

# calculate CDF values
y = 1. * np.arange(len(velocity_sorted)) / (len(velocity_sorted) - 1)

# plot CDF
plt.plot(velocity_sorted, y)
plt.title("CDF of maximum velocity in each frame")
plt.show()

velocity_list = np.array(velocity_list)
print(np.percentile(velocity_list, 99))
print(np.percentile(velocity_list, 1))


# # sort data
# velocity_sorted = np.sort(velocity_list)
#
# # calculate CDF values
# y = 1. * np.arange(len(velocity_sorted)) / (len(velocity_sorted) - 1)
#
# # plot CDF
# plt.plot(velocity_sorted, y)
# plt.show()
# print(v_max)
# print(v_min)

# np.savetxt(os.path.join("../models", "scale_velocity.txt"), np.array([v_min, v_max]))


# syn_doppler = np.load(os.path.join(path, "output/rgb/synth_doppler.npy"))
# sns.heatmap(syn_doppler)
# plt.show()
# sns.heatmap(syn_doppler[400:1000, 5:27])
# plt.show()
# print(1)


# plot fps
# fps = np.loadtxt(os.path.join(path, "frame_rates.txt"))
# plt.plot(np.arange(0, len(fps), 1), fps, label="FPS=120")
# plt.ylabel("fps")
# plt.xlabel("time (s)")
# plt.show()

# doppler_1d = np.load(os.path.join(path, "doppler_gt.npy"))
# sns.heatmap(doppler_1d[1000:1500, :])
# plt.show()

# doppler = np.load(os.path.join(path, "range_doppler_rb94.npy"))
# doppler_1d = np.sum(doppler[:, :, left:right], axis=2)
# hp = sns.heatmap(doppler_1d[2000:3000, :])
# hp.figure.savefig("test.png")
# plt.show()
# doppler_1d = np.sum(doppler[:, :, 17:20], axis=2)
# sns.heatmap(doppler_1d[2000:3000, :])
# plt.show()
# print(1)

