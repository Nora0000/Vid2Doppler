import os
import subprocess



# run locally, not remotely
def scp_visualization_results_all(path='../data/2022_11_09/'):
	# scp all visualization results and rename
	path_list = os.listdir(path)
	i = 0
	for path_i in path_list:
		if path_i == ".DS_Store":
			continue
		path_i = os.path.join(path, path_i)
		for file in os.listdir(path_i):
			if file == ".DS_Store":
				continue
			filename = os.path.join(path_i + '/' + file + '/video_segments/', 'segment_1th_minute_output_signal.mp4')
			source_file= os.path.join('/home/mengjingliu/Vid2Doppler/data/Python', filename)
			dest_file = os.path.join("../data/visualization", filename.split('/')[-4] + '_' + filename.split('/')[-3] + '_' + "segment_1th_minute_output_signal.mp4")
			subprocess.run(['sshpass -p 522166 scp mengjingliu@130.245.191.166:%s %s' % (source_file, dest_file)], shell=True,
	                                    stdout=subprocess.PIPE)
			print(i)
			i += 1


scp_visualization_results_all()
