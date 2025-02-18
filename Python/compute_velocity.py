import argparse
import csv
import math
import cv2
import os
import numpy as np


def main(args):

    # define camera origin position
    camera_orig = [float(i) for i in args.camera_orig[1:-1].split(',')]

    # get video file name
    video_file = args.input_video.split("/")[-1].split(".")[0]

    # get fps of the video
    video = cv2.VideoCapture(args.input_video)
    fps = video.get(cv2.CAP_PROP_FPS)

    # set output flles
    output_path = os.path.join(args.output_folder, os.path.basename(\
                                video_file).replace('.mp4', ''))
    output_path = output_path.replace('.avi', '')
    csv_folder_path = os.path.join(output_path, "frame_velocity/")
    os.makedirs(csv_folder_path, exist_ok=True)

    # get the number of frames
    num_frames = len([name for name in os.listdir(args.output_folder \
                            + video_file + "/frame_position") \
                            if "frame_" in  name])

    # read frame info as numpy arrays from csv files
    vertex_position = []
    vertex_visibilty = []

    if os.path.isfile(output_path + \
                "/../../frames_new.npy"):
        frames = np.load(output_path + \
                    "/../../frames_new.npy", allow_pickle=True)
    else:
        frames = np.load(output_path + \
                    "/../../frames.npy", allow_pickle=True)

    for frame_idx in frames:

        # read frame info for human body
        frame_info = np.genfromtxt(args.output_folder + video_file \
            + "/frame_position/frame_%06d.csv" \
                            % frame_idx, delimiter=',')
        vertex_position.append(frame_info[:, :3])
        vertex_visibilty.append(frame_info[:, 3:])

    # change position and visibility lists to numpy arrays
    vertex_position = np.array(vertex_position)
    vertex_visibilty = np.array(vertex_visibilty)


    # compute radial velocity for human body
    vertex_velocity_list = []
    vertex_distance_list = []


    for frame_idx in range(len(frames)):

        # skip the first frame
        if frame_idx < 1:
            vertex_velocity = np.expand_dims(np.zeros_like(\
                        vertex_position[frame_idx][:,0]), axis=1)
            vertex_velocity_list.append(vertex_velocity)

            p_t_2 = vertex_position[frame_idx] - camera_orig
            mag = np.linalg.norm(p_t_2, axis=1)
            vertex_distance = np.expand_dims(mag, axis=1)
            vertex_distance_list.append(vertex_distance)

        # Calculate radial velocity
        else:

            # compute radial velocity for human body
            v = vertex_position[frame_idx] - vertex_position[frame_idx-1]
            p_t_1 = vertex_position[frame_idx-1] - camera_orig
            p_t_2 = vertex_position[frame_idx] - camera_orig
            v = p_t_2 - p_t_1
            dot_prod = np.multiply(v, p_t_2).sum(axis=1)
            mag = np.linalg.norm(p_t_2, axis=1)
            # vertex_velocity = np.expand_dims(-(dot_prod / mag) * fps, axis=1)   # movement between two frames * frame rate = velocity
            vertex_velocity = np.expand_dims((dot_prod / mag) * fps, axis=1)
            vertex_velocity_list.append(vertex_velocity)

            vertex_distance = np.expand_dims(mag, axis=1)
            vertex_distance_list.append(vertex_distance)


    # scale_vel = np.loadtxt(os.path.join(args.model_path, "scale_velocity.txt"))
    # v_min = scale_vel[0]
    # v_max = scale_vel[1]

    distance_map = np.array(vertex_distance_list)
    distance_map = distance_map[:, :, 0]

    # compute velocity mean for human body using convolution
    velocity_map = np.array(vertex_velocity_list)
    velocity_map = velocity_map[:,:,0]
    for j in range(velocity_map.shape[1]):      # smooth
       # velocity_map[:,j] = np.convolve(velocity_map[:,j], \
       #                          np.ones((5,))/5, mode='same')
       velocity_map[:, j] = np.convolve(velocity_map[:, j], np.ones((5,)) / 5, mode='same')
       # velocity_map[:, j] = np.convolve(velocity_map[:, j], np.ones((10,)) / 10, mode='same')

       distance_map[:, j] = np.convolve(distance_map[:, j], np.ones((5,)) / 5, mode='same')
    velocity_map = np.expand_dims(velocity_map, axis=2)
    distance_map = np.expand_dims(distance_map, axis=2)

    # v_max = max(v_max, np.max(velocity_map[300:-300]))
    # v_min = min(v_min, np.min(velocity_map[300:-300]))

    # np.savetxt(os.path.join(args.model_path, "scale_velocity.txt"), np.array([v_min, v_max]))

    # save velocities and visibilities
    index = 0
    for frame_idx in frames:

        # concatenate velocities and visibilities
        # frame_info = np.concatenate((velocity_map[index], \
                                # vertex_visibilty[index]), axis=1)
        # concatenate velocities, visibilities and distance
        frame_info = np.concatenate((velocity_map[index], vertex_visibilty[index], distance_map[index]), axis=1)

        # save each vertex velocity and visibility
        np.savetxt(csv_folder_path + "frame_%06d.csv" % frame_idx, \
                                            frame_info, delimiter=",")

        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str,
                        help='input video file', default="/home/mengjingliu/Vid2Doppler/data/2023_04_24/2023_04_24_17_34_30_ADL_zx/rgb.avi")

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results', default="/home/mengjingliu/Vid2Doppler/data/2023_04_24/2023_04_24_17_34_30_ADL_zx/output/")

    parser.add_argument('--camera_orig', type=str, default="[0,0,0]",
                        help='camera origin position')      # used to compute the radial projection of velocity

    parser.add_argument('--model_path', type=str, help='Path to DL models', default='../models/')

    args = parser.parse_args()

    main(args)
