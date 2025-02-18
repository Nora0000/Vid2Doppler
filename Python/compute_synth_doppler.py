import numpy as np
import os
from os import listdir
from os.path import isfile, join
from scipy.ndimage import gaussian_filter1d
import argparse
import cv2

N_BINS = 32
DISCARD_BINS = [14, 15, 16, 17]
# N_BINS = 16
# DISCARD_BINS = [6, 7, 8]
GAUSSIAN_BLUR = True
GAUSSIAN_KERNEL = 5
TIME_CHUNK = 1 # 1 second for creating the spectogram


def main(args):

    # get video infomation
    video_file = os.path.basename(args.input_video).replace('.mp4', '')
    video_file = video_file.replace('.avi', '')
    out_path = args.output_folder + '/' + video_file + '/'
    # video = cv2.VideoCapture("./" + args.input_video)
    # fps = video.get(cv2.CAP_PROP_FPS)

    # get frame infomation
    num_frames = len([name for name in \
            os.listdir(out_path + "/frame_velocity") \
            if "frame_" in name])
    output_path = os.path.join(args.output_folder, os.path.basename(\
                                video_file).replace('.mp4', ''))
    output_path = output_path.replace('.avi', '')
    if os.path.isfile(output_path + \
                "/../../frames_new.npy"):
        frames = np.load(output_path + \
                    "/../../frames_new.npy", allow_pickle=True)
    else:
        frames = np.load(output_path + \
                    "/../../frames.npy", allow_pickle=True)
    print("frames: ", num_frames)

    scale_vel = np.loadtxt(os.path.join(args.model_path, "scale_velocity.txt"))
    v_min = scale_vel[0]
    v_max = scale_vel[1]
    v_lim = max(abs(v_max), abs(v_min))

    # compute synthetic doppler data
    synth_doppler_dat = []
    for frame_idx in frames:
        gen_doppler = np.genfromtxt(out_path + "/frame_velocity/frame_%06d.csv" % frame_idx, delimiter=',')
        velocity = gen_doppler[gen_doppler[:, 1]==1, 0]
        distance = gen_doppler[gen_doppler[:, 1]==1, 2]
        # hist = np.histogram(velocity, bins=np.linspace(-2, 2, num=N_BINS+1))[0]     # change velocity range to be consistent with UWB
        # hist = np.histogram(velocity, bins=np.linspace(-1.38, 1.38, num=N_BINS + 1))[0]   # FPS=80
        hist = np.histogram(velocity, bins=np.linspace(-3, 3, num=N_BINS + 1))[0]   #fps=180
        # hist = np.histogram(velocity, bins=np.linspace(-3, 3, num=N_BINS + 1), weights=1 / np.power(distance, 2))[0]
        # hist = np.histogram(velocity, bins=np.linspace(-v_lim, v_lim, num=N_BINS + 1), weights=1 / np.power(distance, 2))[0]

        for bin_idx in DISCARD_BINS:      # ignore stationary parts.
            hist[bin_idx] = 0
        synth_doppler_dat.append(hist/gen_doppler.shape[0])

    synth_doppler_dat = np.array(synth_doppler_dat)

    # if GAUSSIAN_BLUR:           # Don't need Gaussian blur? ignore discard bins in training
    #     for i in range(len(synth_doppler_dat)):
    #         synth_doppler_dat[i] = gaussian_filter1d(synth_doppler_dat[i], GAUSSIAN_KERNEL)

    np.save(out_path+"/synth_doppler.npy", synth_doppler_dat)

    model_path = args.model_path
    scale_vals = np.load(model_path + "scale_vals_new.npy")
    # scale_vals[1] = max(scale_vals[1], np.max(synth_doppler_dat))
    # scale_vals[3] = min(scale_vals[3], np.min(synth_doppler_dat))
    #
    # number = len(synth_doppler_dat)
    # scale_vals[8:11] = (scale_vals[8:11] * scale_vals[11] + np.mean(synth_doppler_dat[:, DISCARD_BINS], axis=0) * number) / (
    #         scale_vals[11] + number)
    # scale_vals[11] += number
    # np.save(os.path.join(model_path, "scale_vals_new.npy"), scale_vals)
    return np.max(synth_doppler_dat), np.min(synth_doppler_dat), scale_vals[8:11]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str, help='input video file',
                        default="/home/mengjingliu/Vid2Doppler/data/2023_05_03/2023_05_03_16_34_39_mengjing_push_diagonal/rgb.avi")

    parser.add_argument('--output_folder', type=str, help='output folder to write results',
                        default="/home/mengjingliu/Vid2Doppler/data/2023_05_03/2023_05_03_16_34_39_mengjing_push_diagonal/output/")

    parser.add_argument('--model_path', type=str, help='Path to DL models', default="../models/")

    args = parser.parse_args()

    main(args)
