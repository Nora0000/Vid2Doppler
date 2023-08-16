import os

import numpy as np
import matplotlib.pyplot as plt

path = "/home/mengjingliu/Vid2Doppler/data/2023_05_04/2023_05_04_18_07_20_mengjing_push/"

path_position = os.path.join(path, "output/rgb/frame_position")

frames = np.load(os.path.join(path, "frames.npy"))

# frame_num = os.listdir(path)
# frame_num = min(len(frame_num), 1200)

# vertex_position_x = np.array([])
# vertex_position_y = np.array([])
# vertex_position_z = np.array([])
vertex_position_x = []
vertex_position_y = []
vertex_position_z = []

cnt = 0
for frame_idx in frames[200:-200]:
	# read frame info for human body
	frame_info = np.genfromtxt(path_position + "/frame_%06d.csv" \
	                           % frame_idx, delimiter=',')
	vertex_position_x.append(np.mean(frame_info[frame_info[:, 3]==1, 0]))
	vertex_position_y.append(np.mean(frame_info[frame_info[:, 3]==1, 1]))
	vertex_position_z.append(np.mean(frame_info[frame_info[:, 3]==1, 2]))
	# vertex_position_x.append(np.mean(frame_info[:, 0]))
	# vertex_position_y.append(np.mean(frame_info[:, 1]))
	# vertex_position_z.append(np.mean(frame_info[:, 2]))

	# if len(vertex_position_z) == 0:
	# 	vertex_position_x = frame_info[frame_info[:, 3]==1, 0]
	# 	vertex_position_y = frame_info[frame_info[:, 3]==1, 1]
	# 	vertex_position_z = frame_info[frame_info[:, 3]==1, 2]
	# else:
	# 	vertex_position_x = np.vstack((vertex_position_x, frame_info[frame_info[:, 3]==1, 0]))
	# 	vertex_position_y = np.vstack((vertex_position_y, frame_info[frame_info[:, 3]==1, 1]))
	# 	vertex_position_z = np.vstack((vertex_position_z, frame_info[frame_info[:, 3]==1, 2]))
	# vertex_position_x.append(frame_info[frame_info[:, 3]==1, 0])
	# vertex_position_y.append(frame_info[frame_info[:, 3]==1, 1])
	# vertex_position_z.append(frame_info[frame_info[:, 3]==1, 2])
	# vertex_visibilty.append(frame_info[:, 3:])
	cnt += 1
	if cnt >= 1200:
		break

vertex_position_x = np.array(vertex_position_x)
pos_mean_x = vertex_position_x
# pos_mean_x = np.mean(vertex_position_x, axis=1)
vertex_position_y = np.array(vertex_position_y)
pos_mean_y = vertex_position_y
# pos_mean_y = np.mean(vertex_position_y, axis=1)
vertex_position_z = np.array(vertex_position_z)
pos_mean_z = vertex_position_z
# pos_mean_z = np.mean(vertex_position_z, axis=1)
mmax = max(np.max(pos_mean_z), np.max(pos_mean_y), np.max(pos_mean_x))
mmin = min(np.min(pos_mean_z), np.min(pos_mean_y), np.min(pos_mean_x))
plt.plot(np.arange(0, cnt, 1), pos_mean_x)
plt.ylabel("x-axis coordination (m)")
plt.xlabel("time")
plt.ylim([mmin, mmax])
plt.show()
plt.plot(np.arange(0, cnt, 1), pos_mean_y)
plt.ylabel("y-axis coordination (m)")
plt.xlabel("time")
plt.ylim([mmin, mmax])
plt.show()
plt.plot(np.arange(0, cnt, 1), pos_mean_z)
plt.ylabel("z-axis coordination (m)")
plt.xlabel("time")
plt.ylim([mmin, mmax])
plt.show()
print(1)
