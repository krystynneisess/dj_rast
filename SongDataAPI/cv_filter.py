import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from song_data_api import SongDataAPI
from hist_match import Equalizer
from contrast import Contrast

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


# It was useful to plot the histogram of the data because sometimes it was concentrated around
# some average value making it always blurred and never in focus. For this feature especially it
# important to center the changes in the image around the average.
def updatefig(fig):
    global a, avg, diff
    a += 1
    value = abs(avg-loudness_spread[a])/diff
    dst = blur(img, value)
    im.set_array(dst)
    return im,

# file = 'tree'
# file = 'clown'
# file = 'guitarist'
# file = 'fox'																																																																																																																																																																
# file = 'shape'
# file = 'chemise'
# file = 'city'
# file = 'parasol'
file = 'under_expose'

song = SongDataAPI("fly_me_to_the_moon")
loudness_spread = song.get_member("loudness_spread", "data")

img = cv2.imread('./images/' + file + '.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

eq_dst = Equalizer(img)
img = eq_dst.equalize_color()

a = 0
avg = np.mean(loudness_spread)
diff = max(abs(avg - np.amax(loudness_spread)), abs(avg - np.amin(loudness_spread)))
fig = plt.figure()
im = plt.imshow(img, animated=True, cmap='Greys_r')
ani = animation.FuncAnimation(fig, updatefig, interval=200, blit=True)

# f = plt.figure(2, figsize=(20,8))
# ax1 = f.add_subplot(121)
# ax2 = f.add_subplot(122)
# # ax3 = f.add_subplot(223)
# # ax4 = f.add_subplot(224)
# # ax1.imshow(img)
# # ax2.hist(img.ravel(), 256, [0, 255])
# # ax3.imshow(dst)
# # ax4.hist(loudness_spread.ravel(), 256, [0, 1])
# ax1.imshow(img)
# ax2.imshow(dst)


plt.show()

