import os
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(input_video):

	begin_time = time.time()
	folder_path = os.path.dirname(os.path.abspath(input_video))
	out_path = folder_path + "/output/"

	# os.system("python3 run_VIBE.py --input_video %s --output_folder %s" % (args.input_video, folder_path))
	# os.system("python3 compute_position.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	# os.system("python3 interpolate_frames.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python3 compute_velocity.py --input_video %s --output_folder %s" % (input_video, out_path))
	os.system("python3 compute_synth_doppler.py --input_video %s --output_folder %s --model_path %s" % (
	input_video, out_path, "../models/"))
	os.system("python3 compute_doppler_gt_new.py --input_video %s" % input_video)
	# os.system("python3 compute_visualization.py --input_video %s --output_folder %s --wireframe" % (args.input_video, out_path))
	# os.system("python3 plot_synth_dop.py --input_video %s --model_path %s --doppler_gt" % (args.input_video, args.model_path))

	# free all temporary memory
	# image_folder = str(np.load(folder_path + "/image_folder.npy"))
	# shutil.rmtree(image_folder)

	print("time cost: {} minutes".format((time.time() - begin_time) / 60))

	doppler_gt = np.load(os.path.join(folder_path, "doppler_gt.npy"))
	doppler_syn = np.load(os.path.join(folder_path, "output/rgb/synth_doppler.npy"))
	frames_common = np.load(os.path.join(folder_path, "frames_common.npy"))
	if os.path.isfile(os.path.join(folder_path, 'frames_new.npy')):
		frames = np.load(os.path.join(folder_path, 'frames_new.npy'), allow_pickle=True)
	else:
		frames = np.load(os.path.join(folder_path, 'frames.npy'), allow_pickle=True)
	indices = np.isin(frames, frames_common)
	return doppler_gt, doppler_syn[indices, :]


def compute_synth_doppler_all(path='../data/2023_05_04/'):
	doppler_gt = []
	doppler_syn = []
	path_list = os.listdir(path)
	i = 0
	for path_i in path_list:
		if path_i.split('_')[-1] == "walk":
			continue
		path_i = os.path.join(os.path.abspath(path), path_i)
		if not os.path.isdir(path_i):
			continue
		gt, syn = main(os.path.join(path_i, "rgb.avi"))
		doppler_gt.append(gt.tolist())
		doppler_syn.append(syn.tolist())
		print(i)
		print(path_i)
		i += 1
	np.save(os.path.join(path, "doppler_gt.npy"), np.array(doppler_gt))
	np.save(os.path.join(path, "doppler_syn.npy"), np.array(doppler_syn))


compute_synth_doppler_all()

# path = '../data/2023_01_18/'
# # compute_synth_doppler_all(path)
# # compute_doppler_gt_all(path)
# # plot_all(path)
# path_list = os.listdir(path)
# i = 0
# for path_i in path_list:
# 	path_i = os.path.join(os.path.abspath(path), path_i)
# 	for file in os.listdir(path_i):
# 		# input_video = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute.mp4')
# 		input_video = os.path.join(path_i + '/' + file, 'rgb.avi')
# 		main("", input_video=input_video)
# 		print(i)
# 		print(input_video)
# 		i += 1
