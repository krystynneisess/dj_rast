import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class Contrast:

	def __init__(self, img_data = None):
		self.data = img_data

	def dist(self, alpha, beta) : 
		dyn = np.empty(256, dtype=np.float32)
		for i in range(256): 
			dyn[i] = min(i * alpha + beta, 255)
		return dyn

	def contrast(self, alpha, beta) :
		size = self.data.shape
		height = size[0]
		width = size[1]
		dyn = self.dist(alpha, beta)
		if len(size) == 3: 
			dst = np.empty((height, width, 3), dtype=np.float32)
			for i in range(height): 
				for j in range(width):
					dst[i, j, 0] = dyn[self.data[i, j, 0]]
					dst[i, j, 1] = dyn[self.data[i, j, 1]]
					dst[i, j, 2] = dyn[self.data[i, j, 2]]
		else: 
			dst = np.empty((height, width), dtype=np.float32)
			for i in range(height): 
				for j in range(width): 
					dst[i, j] = dyn[self.data[i, j]]
		return np.array(dst, dtype=np.uint8)

# def contrast(img, value) :
# 	f = 259.0*(value + 255.0) / (255.0*(259.0 - value))
# 	# print(f)
# 	# img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# 	height, width, x = img.shape
# 	dst = np.empty((height, width, 3), dtype=np.float32)
# 	for i in range(height):
# 		for j in range(width):
# 			dst[i, j, 0] = m.floor(f*(img[i, j, 0] - 128) + 128)
# 			dst[i, j, 1] = m.floor(f*(img[i, j, 1] - 128) + 128)
# 			dst[i, j, 2] = m.floor(f*(img[i, j, 2] - 128) + 128)
# 	return dst