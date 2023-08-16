import subprocess
import cv2
import math
import os


# not work with '.avi'. can work with '.mp4'
def get_duration_from_cv2(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        return duration     # number of seconds
    return -1


def seg_video_by_minute(filename):
    path = os.path.dirname(os.path.abspath(filename))
    path = path + '/video_segments/'
    if not os.path.exists(path):
        os.mkdir(path)
    out_filename = 'segment_{}th_minute.mp4'.format(1)   # 1 minute
    out_filename = os.path.join(path, out_filename)
    result = subprocess.run(['ffmpeg -ss 00:00:00 -i "%s" -c copy -t 60 "%s"' % (filename, out_filename)],
                            shell=True, stdout=subprocess.PIPE)
    return 0
    # duration = get_duration_from_cv2(filename)
    # if duration > 0:
    #     minutes = math.floor(duration / 60)
    #     for i in range(minutes):
    #         if i < 10:
    #             start = '00:0{}:00'.format(i)
    #         else:
    #             start = '00:{}:00'.format(i)
    #         out_filename = 'segment_{}.mp4'.format(i)
    #         out_filename = os.path.join(path, out_filename)
    #         result = subprocess.run(['ffmpeg -ss "%s" -i "%s" -c copy -t 60 "%s"' % (start, filename, out_filename)], shell=True,
    #                                 stdout=subprocess.PIPE)
    #         print(result)
    #     return 0
    # return -1


if __name__ == "__main__":
    path = os.path.abspath('../data/2022_11_09/har8/')
    dires = os.listdir(path)
    for dire in dires:
        filename = os.path.join(path + '/' + dire, 'rgb.avi')
        seg_video_by_minute(filename)

