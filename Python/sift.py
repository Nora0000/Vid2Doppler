import numpy as np
import cv2
# from psd_tools import PSDImage
from PIL import Image
import os
from matplotlib import pyplot as plt
import seaborn as sns

# path4 = "/home/mengjingliu/Vid2Doppler/data/2023_07_19/HAR6"
path4 = "/Users/liumengjing/Documents/HAR/Vid2Doppler/data/2023_07_19/HAR6"
x4 = np.load(os.path.join(path4, "X_4.npy"))
x4_syn = np.load(os.path.join(path4, "X_4_syn.npy"))
y4 = np.load(os.path.join(path4, "Y_4.npy"))

def sift_feature(d1):
	d1 = (d1 - np.min(d1)) / (np.max(d1) - np.min(d1))
	# m1 = Image.fromarray(np.uint8(d1 * 255), 'L')
	m1 = np.uint8(d1 * 255)
	
	sift = cv2.xfeatures2d.SIFT_create()
	
	kp1, des1 = sift.detectAndCompute(m1, None)
	return kp1, des1, m1


def sift_compare(d1, d2):
	kp1, des1, m1 = sift_feature(d1)
	kp2, des2, m2 = sift_feature(d2)
	
	# sns.heatmap(x4_syn[i, :, :])
	# plt.show()
	
	img1 = cv2.drawKeypoints(m1, kp1, m1)
	plt.imshow(img1)
	plt.show()
	# cv2.imshow('image1', img1)
	img2 = cv2.drawKeypoints(m2, kp2, m2)
	# cv2.imshow('image2', img2)
	plt.imshow(img2)
	plt.show()
	
	flann_index_kdtree = 1
	index_params = dict(algorithm=flann_index_kdtree, trees=5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)
	goodMatch = []

	for m, n in matches:
		if m.distance < 0.5 * n.distance:
			goodMatch.append(m)

	goodMatch = np.expand_dims(goodMatch, 1)

	print(goodMatch[: 20])

	# img_out = cv2.drawMatchesKnn(m1, kp1, m2, kp2, goodMatch[: 15])
	for i in range(len(matches)):
		img_out = cv2.drawMatchesKnn(m1, kp1, m2, kp2, [matches[i]], None)
		plt.imshow(img_out)
		plt.show()
	# cv2.imwrite(os.path.join(path4, "{}.jpg".format(i)), img_out)
	# cv2.imshow('image', img_out)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


for i in range(len(x4)):
	d1 = x4[i, :, :]
	d1 = np.uint8((d1 - np.min(d1)) / (np.max(d1) - np.min(d1)) * 255)
	# m1 = Image.fromarray(np.uint8(d1 * 255), 'L')
	# gray = cv2.cvtColor(np.uint8(d1 * 255), cv2.COLOR_BGR2GRAY)  # 转化为灰度图
	ret, thresh = cv2.threshold(d1, 127, 255, cv2.THRESH_BINARY)  # 阈值二值化
	contours, hierarchy = cv2.findContours(np.uint8(thresh/255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # 寻找轮廓
	img2 = cv2.drawContours(d1, contours, -1, (255, 0, 0), 2)
	plt.imshow(img2)
	plt.show()
	
	d1 = x4_syn[i, :, :]
	d1 = np.uint8((d1 - np.min(d1)) / (np.max(d1) - np.min(d1)) * 255)
	# m1 = Image.fromarray(np.uint8(d1 * 255), 'L')
	# gray = cv2.cvtColor(np.uint8(d1 * 255), cv2.COLOR_BGR2GRAY)  # 转化为灰度图
	ret, thresh = cv2.threshold(d1, 127, 255, cv2.THRESH_BINARY)  # 阈值二值化
	contours, hierarchy = cv2.findContours(np.uint8(thresh / 255), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # 寻找轮廓
	img2 = cv2.drawContours(d1, contours, -1, (255, 0, 0), 2)
	plt.imshow(img2)
	plt.show()
	
	

