import os

path = '../data/2022_11_09/'
path_list = os.listdir(path)
i = 0
for path_i in path_list:
	path_i = os.path.join(os.path.abspath(path), path_i)
	for file in os.listdir(path_i):
		# input_video = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute.mp4')
		input_video = os.path.join(path_i + '/' + file, 'rgb0.avi')
		os.rename(input_video, path_i + '/' + file + '/' + 'rgb.avi')
		print(i)
		print(input_video)
		i += 1
