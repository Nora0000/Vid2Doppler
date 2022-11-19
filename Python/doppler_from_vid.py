import argparse
import os
import random
import math
import time
import numpy as np
import shutil


def main(args):

	begin_time = time.time()
	folder_path = os.path.dirname(os.path.abspath(args.input_video))
	os.system("python run_VIBE.py --input_video %s --output_folder %s" % (args.input_video, folder_path))

	out_path = folder_path + "/output/"

	os.system("python compute_position.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python interpolate_frames.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python compute_velocity.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	if args.visualize_mesh:
		os.system("python compute_visualization.py --input_video %s --output_folder %s --wireframe" % (args.input_video, out_path))
	os.system("python compute_synth_doppler.py --input_video %s --output_folder %s --model_path %s" % (args.input_video, out_path, args.model_path))
	# if args.doppler_gt:
	# 	os.system("python plot_synth_dop.py --input_video %s --model_path %s --doppler_gt" % (args.input_video, args.model_path))
	# else:
	# 	os.system("python compute_doppler_gt.py --input_video %s" % args.input_video)
	# 	os.system("python plot_synth_dop.py --input_video %s --model_path %s --doppler_gt" % (args.input_video, args.model_path))

	# free all temporary memory
	image_folder = str(np.load(folder_path + "/image_folder.npy"))
	shutil.rmtree(image_folder)

	print("time cost: {} minutes".format((time.time() - begin_time) / 60))


def doppler_from_vid_all():
	path = '../data/2022_11_09/'
	path_list = os.listdir(path)
	i = 0
	for path_i in path_list:
		path_i = os.path.join(os.path.abspath(path), path_i)
		for file in os.listdir(path_i):
			input_video = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute.mp4')
			os.system("python doppler_from_vid.py --input_video %s --model_path %s" % (input_video, '../models/'))
			print(i)
			i += 1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file')

	parser.add_argument('--visualize_mesh', help='Render visibility mesh and velocity map', action='store_true')

	parser.add_argument('--model_path', type=str, help='Path to DL models')

	parser.add_argument('--doppler_gt', help='Doppler Ground Truth is available for reference', action='store_true')

	args = parser.parse_args()

	main(args)
