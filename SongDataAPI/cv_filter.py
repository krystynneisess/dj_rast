import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math as m
from song_data_api import SongDataAPI
from hist_match import Equalizer
from edge_detect import EdgeDetect 
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

# It was useful to plot the histogram of the data because sometimes it was concentrated around
# some average value making it always blurred and never in focus. For this feature especially it
# important to center the changes in the image around the average.
def updatefig(fig):
    global a, avg, diff
    a += 10
    value = abs(avg-loudness_spread[a])/diff
    dst = con_img.contrast(1.0+value, value*20)
    # print(value)
    # print(str(value*20))
    # dst = blur(img, value)
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
# file = 'lena_low'
# file = 'test'

song = SongDataAPI("fly_me_to_the_moon")
loudness_spread = song.get_member("loudness_spread", "data")

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
avg = np.mean(loudness_spread)
diff = max(abs(avg - np.amax(loudness_spread)), abs(avg - np.amin(loudness_spread)))
fig = plt.figure()
im = plt.imshow(img, animated=True)
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

# f = plt.figure()
# ax1 = f.add_subplot(121)
# ax2 = f.add_subplot(122)
# ax1.imshow(img)
# ax2.imshow(dst)


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

