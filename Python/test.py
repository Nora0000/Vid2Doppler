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
from helper import color_scale, chose_central_rangebins
import matplotlib

path = "../data/2023_05_02/2023-05-03-17-16-46_mengjing_zigzag_diagonal_fps180"

# plot fps
# fps = np.loadtxt(os.path.join(path, "frame_rates.txt"))
# plt.plot(np.arange(0, len(fps), 1), fps, label="FPS=120")
# plt.ylabel("fps")
# plt.xlabel("time (s)")
# plt.show()

# doppler_1d = np.load(os.path.join(path, "doppler_gt.npy"))
# sns.heatmap(doppler_1d[1000:1500, :])
# plt.show()

doppler = np.load(os.path.join(path, "range_doppler_rb94.npy"))
doppler_1d = np.sum(doppler, axis=2)
sns.heatmap(doppler_1d[2000:3000, :])
plt.show()
doppler_1d = np.sum(doppler[:, :, :14], axis=2)
sns.heatmap(doppler_1d[2000:3000, :])
plt.show()
print(1)

