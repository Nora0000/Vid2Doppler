import argparse
import os
import random
import math
import time
import numpy as np
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):

	begin_time = time.time()
	folder_path = os.path.dirname(os.path.abspath(args.input_video))
	out_path = folder_path + "/output/"

	os.system("python3 run_VIBE.py --input_video %s --output_folder %s" % (args.input_video, folder_path))
	os.system("python3 compute_position.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	# os.system("python3 interpolate_frames.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python3 compute_velocity.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python3 compute_synth_doppler.py --input_video %s --output_folder %s --model_path %s" % (args.input_video, out_path, args.model_path))
	os.system("python3 compute_doppler_gt_new.py --input_video %s" % args.input_video)
	# os.system("python3 compute_visualization.py --input_video %s --output_folder %s --wireframe" % (args.input_video, out_path))
	os.system("python3 plot_synth_dop.py --input_video %s --model_path %s --doppler_gt" % (args.input_video, args.model_path))

	# free all temporary memory
	image_folder = str(np.load(folder_path + "/image_folder.npy"))
	shutil.rmtree(image_folder)

	print("time cost: {} minutes".format((time.time() - begin_time) / 60))

	# doppler_gt = np.load(os.path.join(folder_path, "doppler_gt.npy"))
	# doppler_syn = np.load(os.path.join(folder_path, "output/rgb/synth_doppler.npy"))
	# frames_common = np.load(os.path.join(folder_path, "frames_common.npy"))
	# if os.path.isfile(os.path.join(folder_path, 'frames_new.npy')):
	# 	frames = np.load(os.path.join(folder_path, 'frames_new.npy'), allow_pickle=True)
	# else:
	# 	frames = np.load(os.path.join(folder_path, 'frames.npy'), allow_pickle=True)
	# indices = np.isin(frames, frames_common)
	# return doppler_gt, doppler_syn[indices, :]


def compute_synth_doppler_all():
	path = '../data/2022_11_09/'
	path_list = os.listdir(path)
	i = 0
	for path_i in path_list:
		path_i = os.path.join(os.path.abspath(path), path_i)
		for file in os.listdir(path_i):
			# input_video = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute.mp4')
			input_video = os.path.join(path_i + '/' + file, 'rgb.avi')
			if os.path.exists(os.path.join(path_i + '/' + file + '/output/rgb/', 'synth_doppler.npy')):
				continue
			os.system("python doppler_from_vid.py --input_video %s --model_path %s" % (input_video, '../models/'))
			print(i)
			i += 1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file',
	                    default="/home/mengjingliu/Vid2Doppler/data/2023_11_17/HAR3/2023_11_17_12_34_31_bend/rgb.avi")

	parser.add_argument('--visualize_mesh', help='Render visibility mesh and velocity map', action='store_true')

	parser.add_argument('--model_path', type=str, help='Path to DL models', default='../models/')

	parser.add_argument('--doppler_gt', help='Doppler Ground Truth is available for reference', action='store_true')

	args = parser.parse_args()



	main(args)
