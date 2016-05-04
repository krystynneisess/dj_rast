import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from song_data_api import SongDataAPI
from hist_match import Equalizer
from edge_detect import EdgeDetect 
from contrast import Contrast
from scipy.interpolate import UnivariateSpline


# still to do: contrast, brightness, painterly effects
def convolve(img, kernel) :
	k_height, k_width = kernel.shape
	nk = np.empty((k_height, k_width), dtype=np.float32)

	off = (k_height-1)/2
	for a in range(k_height):
		for b in range(k_width):
			nk[k_height-1-a, k_width-1-b] = kernel[a, b]
	dim = img.shape
	i_height = dim[0]
	i_width = dim[1]

	if len(img.shape) == 3: 
		dst = np.empty((i_height, i_width, 3), dtype=np.uint8)
		for i in range(off, i_height-off):
			for j in range(off, i_width-off):
				sum0 = 0.0
				sum1 = 0.0
				sum2 = 0.0
				for ii in range(k_height):
					for jj in range(k_width):
						sum0 += nk[ii, jj]*img[i-off+ii, j-off+jj, 0]
						sum1 += nk[ii, jj]*img[i-off+ii, j-off+jj, 1]
						sum2 += nk[ii, jj]*img[i-off+ii, j-off+jj, 2]
				dst[i, j, 0] = sum0
				dst[i, j, 1] = sum1
				dst[i, j, 2] = sum2
	else:
		dst = np.empty((i_height, i_width), dtype=np.uint8)
		for i in range(off, i_height-off):
			for j in range(off, i_width-off):
				sum0 = 0.0
				for ii in range(k_height):
					for jj in range(k_width):
						sum0 += nk[ii, jj]*img[i-off+ii, j-off+jj]
				dst[i, j] = sum0
	return dst


def sharpen(img, value) : 
	factor = 10 * value
	n_factor = -factor / 4
	kernel = np.array([[0, n_factor, 0], [n_factor, factor, n_factor], [0, n_factor, 0]]) # sharpen
	return cv2.filter2D(img, -1, kernel)

# Takes in an image and value [0,1]. Blurs the image according to the value (high value -> more blur)
def blur(img, value) :
	factor = m.floor(value*20)
	if (factor == 0) :
		factor = 1
	kernel = np.ones((factor,factor),np.float32)/pow(factor, 2) # blur
	return cv2.filter2D(img, -1, kernel)


def edge_detect(img, value) :
	vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #  vertical gradient
	horizontal = np.array([[-1, -2, -2], [0, 0, 0], [1, 2, 2]]) # horizontal gradient
	if value == 0 or value == 1 or value == 2 or value == 6: # gray result
		gray = len(img.shape)
		if (gray == 3) : 
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img2 = cv2.bilateralFilter(img, 15, 80, 80)

	img1 = np.array(img2, dtype=np.float64)

	v_grad = cv2.filter2D(img1, -1, vertical)
	h_grad = cv2.filter2D(img1, -1, horizontal)

	abs_v = np.absolute(v_grad)
	abs_h = np.absolute(h_grad)

	v_grad = np.array(abs_v, dtype=np.uint8)
	h_grad = np.array(abs_h, dtype=np.uint8)
	if value == 1 or value == 4:
		return v_grad
	elif value == 2 or value == 5: 
		return h_grad
	else: 	
		if value == 3: #color image
			height, width, x = img.shape
			dst = np.empty((height, width, 3), dtype=np.uint8)
			for i in range(height):
				for j in range(width): 
					sum0 = pow(v_grad[i, j, 0], 2) + pow(h_grad[i, j, 0], 2)
					sum1 = pow(v_grad[i, j, 1], 2) + pow(h_grad[i, j, 1], 2)
					sum2 = pow(v_grad[i, j, 2], 2) + pow(h_grad[i, j, 2], 2)
					dst[i, j, 0] = pow(sum0, .5)
					dst[i, j, 1] = pow(sum1, .5)
					dst[i, j, 2] = pow(sum2, .5)
			dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
		else: 
			height, width = img.shape
			dst = np.empty((height, width), dtype=np.uint8)
			#edge detection grayscale:
			for i in range(height):
				for j in range(width):
					sum0 = pow(v_grad[i, j], 2) + pow(h_grad[i, j], 2)
					dst[i, j] = pow(sum0, .5)
			if value == 6: 
				dst = blur(dst, .7)
				dstv = cv2.filter2D(dst, -1, vertical)
				dsth = cv2.filter2D(dst, -1, horizontal)
				for i in range(height):
					for j in range(width):
						sum0 = pow(dstv[i, j], 2) + pow(dsth[i, j], 2)
						dst[i, j] = pow(sum0, .5)
		return dst

def brightness(in_img, amount):

	img = np.copy(in_img)
	height, width, x = img.shape
	value = (amount - .5)*2*255
	arr = create_LUT_manual(value)

	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	for i in range(0, height):
		for j in range(0, width):
			img[i, j, 2] = arr[img[i, j, 2]]

	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	return img

def invert(img, value):
	#img = np.copy(in_img)
	height, width, x = img.shape
	arr = create_LUT_invert();

	for i in range(0, height):
		for j in range(0, width):
			img[i, j, 0] = arr[img[i, j, 0]]
			img[i, j, 1] = arr[img[i, j, 1]]
			img[i, j, 2] = arr[img[i, j, 2]]
	return img

def saturate(in_img, amount):
    img = np.copy(in_img)	
    height, width, x = img.shape
    value = (amount - .5)*2*255
    arr = create_LUT_manual(value)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    for i in range(0, height):
    	for j in range(0, width):
    		img[i, j, 1] = arr[img[i, j, 1]]

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img
    
def create_LUT_8UC1(x, y):
	spl = UnivariateSpline(x, y)
	return spl(xrange(256))

def create_LUT_manual(amount):
	arr = []
	for i in range(256):
		arr.append(min(max(0, i + amount), 255))
	return arr

def create_LUT_invert():
	arr = []
	for i in range(256):
		arr.append(abs(i - 255))
	return arr

def temperature(img, amount):
	if (amount < 0 or amount > 1):
		raise NameError('amount must be between 0 and 1')
	elif amount >= .5:
		return warmer(img, (amount-.5)*2)
	else:
		return cooler(img, (.5 - amount)*2)

def warmer(in_img, amount):
	img = np.copy(in_img)

	incr_ch_lut = create_LUT_8UC1([0, 64,      192,      256],
	                              [0, 64 + 40*amount, 192 + 45*amount, 256])
	decr_ch_lut = create_LUT_8UC1([0, 64,      192,      256],
                       	          [0, 64 - 52*amount, 192 - 85*amount, 192])
	 
	c_r, c_g, c_b = cv2.split(img)
	c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
	c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
	img = cv2.merge((c_r, c_g, c_b))

	c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
	 

	c_h, c_s, c_v = cv2.split(cv2.cvtColor(img,
	    cv2.COLOR_RGB2HSV))
	c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
	 
	img_warmer = cv2.cvtColor(cv2.merge(
	                      (c_h, c_s, c_v)),
	                       cv2.COLOR_HSV2RGB)
	return img_warmer

def cooler(in_img, amount):
	img = np.copy(in_img)

	incr_ch_lut = create_LUT_8UC1([0, 64,      192,      256],
	                              [0, 64 + 40*amount, 192 + 45*amount, 256])
	decr_ch_lut = create_LUT_8UC1([0, 64,      192,      256],
                       	          [0, 64 - 52*amount, 192 - 85*amount, 192])

	c_r, c_g, c_b = cv2.split(img)
	c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
	c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
	img = cv2.merge((c_r, c_g, c_b))

	c_h, c_s, c_v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
	c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
	img_cooler = cv2.cvtColor(cv2.merge(
    	                     (c_h, c_s, c_v)), 
                              cv2.COLOR_HSV2RGB)
	return img_cooler