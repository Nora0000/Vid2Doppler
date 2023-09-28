import os
import numpy as np
import matplotlib
from helper import get_spectograms, root_mean_squared_error, color_scale, rolling_window_combine
from tensorflow.keras.models import load_model
import pickle
import cv2
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args, input_video="", model_path="", if_doppler_gt=False):
	if model_path == "":
		model_path = args.model_path

	# lb = pickle.loads(open(model_path + "classifier_classes.lbl", "rb").read())
	# autoencoder = load_model(model_path + "autoencoder_weights.hdf5",
	#                          custom_objects={'root_mean_squared_error': root_mean_squared_error})
	# scale_vals = np.load(model_path + "scale_vals_new.npy")
	fps = 24
	TIME_CHUNK = 3
	# max_dopVal = scale_vals[2]
	# max_synth_dopVal = scale_vals[0]
	# min_dopVal = scale_vals[3]
	# min_synth_dopVal = scale_vals[1]
	# mean_discard_bins = scale_vals[4:7]
	# mean_synth_discard_bins = scale_vals[8:11]
	DISCARD_BINS = [14, 15, 16, 17]
	bin_num=32
	# DISCARD_BINS = [6, 7, 8]

	if input_video == "":
		vid_f = args.input_video
	else:
		vid_f = input_video
	in_folder = os.path.dirname(vid_f)
	vid_file_name = vid_f.split("/")[-1].split(".")[0]

	cap = cv2.VideoCapture(vid_f)

	frames_common = np.load(os.path.join(in_folder, 'frames_common.npy')).astype(int)
	if os.path.isfile(os.path.join(in_folder, 'frames_new.npy')):
		frames = np.load(os.path.join(in_folder, 'frames_new.npy'), allow_pickle=True)
	else:
		frames = np.load(os.path.join(in_folder, 'frames.npy'), allow_pickle=True)
	indices = np.isin(frames, frames_common)
	synth_doppler_dat_f = in_folder + "/output/" + vid_file_name + "/synth_doppler.npy"
	synth_doppler_dat = np.load(synth_doppler_dat_f)[indices, :]
	# synth_doppler_dat = np.load(synth_doppler_dat_f)
	# ll = len(synth_doppler_dat_f)
	# synth_doppler_dat[:, DISCARD_BINS] -= mean_synth_discard_bins
	synth_spec_pred = get_spectograms(synth_doppler_dat, TIME_CHUNK, fps, synthetic=True, zero_pad=True)
	synth_spec_pred = synth_spec_pred.astype("float32")
	synth_spec_test = synth_spec_pred
	# synth_spec_test = (synth_spec_pred - min_synth_dopVal) / (max_synth_dopVal - min_synth_dopVal)

	dop_spec_test = np.zeros_like(synth_spec_test)
	if if_doppler_gt or args.doppler_gt:
		# doppler_dat_pos_f = in_folder + "/../" + "/doppler_gt.npy"
		doppler_dat_pos_f = in_folder + "/doppler_gt.npy"
		doppler_dat_pos = np.load(doppler_dat_pos_f)
		# doppler_dat_pos[:, DISCARD_BINS] -= mean_discard_bins
		doppler_dat_pos[:, DISCARD_BINS] = np.zeros((doppler_dat_pos.shape[0], len(DISCARD_BINS)))
		dop_spec = get_spectograms(doppler_dat_pos, TIME_CHUNK, fps, bin_num, zero_pad=True)
		dop_spec = dop_spec.astype("float32")
		dop_spec_test = dop_spec
		# dop_spec_test = (dop_spec - min_dopVal) / (max_dopVal - min_dopVal)

	# synth_spec_in = (synth_spec_pred - min_synth_dopVal) / (max_synth_dopVal - min_synth_dopVal)
	# decoded = autoencoder.predict(synth_spec_in[:, :, :, np.newaxis])
	# decoded = decoded[:, :, :, 0]
	# decoded[DISCARD_BINS, :] -= mean_discard_bins[:, np.newaxis]
	# decoded = rolling_window_combine(decoded)
	#
	# y_max = max(np.max(decoded), np.max(synth_spec_test), np.max(dop_spec_test))
	# norm = matplotlib.colors.Normalize(vmin=0, vmax=y_max)

	video_idx = 0
	ret, frame = cap.read()
	# frames_common = frames_common[240:-240]
	for idx in range(240, len(frames_common)-240):
	# for idx in frames_common:
		while video_idx != frames_common[idx]:
			ret, frame = cap.read()
			video_idx += 1
		original_synth = color_scale(synth_spec_test[idx],
		                             matplotlib.colors.Normalize(vmin=np.min(synth_spec_test), vmax=np.max(synth_spec_test)),
		                             "Initial Synthetic Doppler")
		od = dop_spec_test[idx]
		# od[14:17, :] = np.zeros((3, 72))

		original_dop = color_scale(od, matplotlib.colors.Normalize(vmin=np.min(dop_spec_test), vmax=np.max(dop_spec_test)),
		                           "Real World Doppler")
		# recon = color_scale(decoded[idx], matplotlib.colors.Normalize(vmin=np.min(decoded), vmax=np.max(decoded)),
		#                     "Final Synthetic Doppler")
		in_frame = color_scale(frame, None, "Input Video")
		output = np.hstack([in_frame, original_dop, original_synth])
		# output = np.hstack([in_frame, original_dop, original_synth, recon])
		if idx == 240:
			height, width = output.shape[0], output.shape[1]
			out_vid = cv2.VideoWriter(in_folder + '/' + vid_file_name + '_output_signal.mp4',
			                          cv2.VideoWriter_fourcc(*'mp4v'),
			                          fps, (width, height))
		out_vid.write(output)
	# cv2.imshow("Recording video stream", output)

	cap.release()
	out_vid.release()


def plot_all(path='../data/2022_11_09/'):
	path_list = os.listdir(path)
	i = 0
	for path_i in path_list:
		path_i = os.path.join(os.path.abspath(path), path_i)
		for file in os.listdir(path_i):
			input_video = os.path.join(path_i + '/' + file, 'rgb.avi')
			# input_video = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute.mp4')
			main("", input_video=input_video, model_path='../models/', if_doppler_gt=True)
			print(i)
			i += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file',
	                    default="/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR5/2023_07_19_21_16_09_push/rgb.avi")

	parser.add_argument('--model_path', type=str, help='Path to DL models', default='../models/')

	parser.add_argument('--doppler_gt', help='Doppler Ground Truth is available for reference', action='store_true',
	                    default=True)

	args = parser.parse_args()

	main(args)
