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

<<<<<<< HEAD
=======
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

def brightness(img, amount):

	height, width, x = img.shape
	value = (amount - .5)*2*255

	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	cv2.imshow('img', img)

	for i in range(0, height):
		for j in range(0, width):
			img[i, j, 2] = min(max(0, img[i, j, 2] + value), 255)

	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	return img

def invert(img, value):
	height, width, x = img.shape

	for i in range(0, height):
		for j in range(0, width):
			img[i, j, 0] = abs(img[i, j, 0] - 255)
			img[i, j, 1] = abs(img[i, j, 1] - 255)
			img[i, j, 2] = abs(img[i, j, 2] - 255)
	return img

def saturate(img, amount):	
	height, width, x = img.shape
	value = (amount - .5)*2*255
	print value
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	for i in range(0, height):
		for j in range(0, width):
			img[i, j, 1] = min(max(0, img[i, j, 1] + value), 255)

	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	return img

def create_LUT_8UC1(x, y):
	spl = UnivariateSpline(x, y)
	return spl(xrange(256))

def temperature(img, amount):
	if (amount < 0 or amount > 1):
		raise NameError('amount must be between 0 and 1')
	elif amount >= .5:
		return warmer(img, (amount-.5)*2)
	else:
		return cooler(img, (.5 - amount)*2)

def warmer(img, amount):

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

def cooler(img, amount):

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


>>>>>>> a8bf8ba6aea668e8c7fd310e0a627b8ef087b5d3
# It was useful to plot the histogram of the data because sometimes it was concentrated around
# some average value making it always blurred and never in focus. For this feature especially it
# important to center the changes in the image around the average.
def updatefig(fig):
    global a, avg, diff
<<<<<<< HEAD
    a += 10
    value = abs(avg-loudness_spread[a])/diff
    dst = con_img.contrast(1.0+value, value*20)
    # print(value)
    # print(str(value*20))
    # dst = blur(img, value)
=======
    a += 1
    value = frequency_spread[a]
    dst = saturate(img, value)
>>>>>>> a8bf8ba6aea668e8c7fd310e0a627b8ef087b5d3
    im.set_array(dst)
    return im,



# file = 'tree'
# file = 'clown'
# file = 'guitarist'
# file = 'fox'																																																																																																																																																																
# file = 'shape'
file = 'chemise'
# file = 'city'
# file = 'parasol'
# file = 'under_expose'
<<<<<<< HEAD
# file = 'lena_low'
# file = 'test'
=======
file = 'birds'
>>>>>>> a8bf8ba6aea668e8c7fd310e0a627b8ef087b5d3

song = SongDataAPI("fly_me_to_the_moon")
frequency_spread = song.get_member("frequency_spread", "data")
frequency_centroid = song.get_member("frequency_centroid", "data")
frequency_centroid = frequency_centroid[50:6000]
frequency_centroid = frequency_centroid/np.max(frequency_centroid)



img = cv2.imread('./images/' + file + '.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Edge Detection block
# e_d_img = EdgeDetect(img)
# dst = e_d_img.detect_edge(3)
# dst_1 = np.array(dst, dtype=np.float32)
# img_1 = np.array(img, dtype=np.float32)
# dst1 = np.array(((dst_1 + img_1)/2), dtype=np.uint8)
# eq_dst1 = Equalizer(dst1)
# eq_img = eq_dst1.equalize_color()

# Contrast Block
con_img = Contrast(img)
dst = con_img.contrast(1, 80)

a = 0
avg = np.mean(frequency_spread)
diff = max(abs(avg - np.amax(frequency_spread)), abs(avg - np.amin(frequency_spread)))
fig = plt.figure()
<<<<<<< HEAD
im = plt.imshow(img, animated=True)
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

# f = plt.figure()
# ax1 = f.add_subplot(121)
# ax2 = f.add_subplot(122)
# ax1.imshow(img)
# ax2.imshow(dst)
=======
im = plt.imshow(img, animated=True, cmap='Greys_r')
ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)

# f = plt.figure(2, figsize=(20,8))
# ax1 = f.add_subplot(221)
# ax2 = f.add_subplot(222)
# diff = max(np.min(frequency_spread) - np.mean(frequency_spread), np.max(frequency_spread) - np.mean(frequency_spread))
# dst = abs(frequency_spread - np.mean(frequency_spread))/diff
# ax3 = f.add_subplot(223)
# ax4 = f.add_subplot(224)
# # ax3 = f.add_subplot(223)
# # ax4 = f.add_subplot(224)
# # ax1.imshow(img)
# # ax2.hist(img.ravel(), 256, [0, 255])
# # ax3.imshow(dst)
# ax1.hist(frequency_spread.ravel(), 256, [0, 1])
# ax2.hist(dst.ravel(), 256, [0, 1])
# ax3.plot(frequency_centroid)
# ax4.plot(frequency_spread)

# # ax1.imshow(img)
# # ax2.imshow(dst)
>>>>>>> a8bf8ba6aea668e8c7fd310e0a627b8ef087b5d3


# f = plt.figure(2, figsize=(20,8))
# ax1 = f.add_subplot(221)
# ax2 = f.add_subplot(222)
# ax3 = f.add_subplot(223)
# ax4 = f.add_subplot(224)
# ax1.imshow(img, cmap='Greys_r')
# ax3.imshow(dst, cmap='Greys_r')
# ax2.hist(img.ravel(), 256, [0, 255])
# ax4.hist(dst.ravel(), 256, [0, 255])

plt.show()

