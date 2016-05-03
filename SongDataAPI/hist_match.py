import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.animation as animation
import math

# create the new distribution of pixels
class Equalizer:

	def __init__(self, img_data = None):
		self.data = img_data

	def dist(self, hist, total) :
		dyn = np.empty(256, dtype=np.float32)
		sumn = 0.0
		sumn_1 = 0.0
		for i in range(256):
			sumn = hist[i] + sumn_1
			dyn[i] = math.floor((255)*sumn/total)
			sumn_1 = sumn
		return dyn

	# def prob_n(n , hist, total) :
	# 	return hist[n]/total

	def equalize_gray(self): 
		height, width = self.data.shape
		# if (width == 1) :
		# 	hist = np.histogram(self.data, 256, (0, 1))
		# 	# hist = cv2.calcHist([self.data],[0],None,[256],[0,1])
		# else:
			# hist = np.histogram(self.data, 256, (0, 256))
		hist = cv2.calcHist([self.data],[0],None,[256],[0,256])

		height, width = self.data.shape
		total = height * width
		dyn = self.dist(hist, total)
		dst = np.empty((height, width), dtype=np.float32)
		for i in range(height): 
			for j in range(width):
				pix = self.data[i, j]
				dst[i, j] = dyn[pix]
		return dst

	def equalize_color(self):
		img1 = cv2.cvtColor(self.data, cv2.COLOR_RGB2HSV)
		dst = img1[:, :, 2]
		val = Equalizer(dst)
		eq = val.equalize_gray() 
		img1[:, :, 2] = eq
		return cv2.cvtColor(img1, cv2.COLOR_HSV2RGB)
