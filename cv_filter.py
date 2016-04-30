import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.animation as animation

# still to do: contrast, brightness, painterly effects

def sharpen(img, value) : 
	factor = 10 * value
	n_factor = -factor / 4
	kernel = np.array([[0, n_factor, 0], [n_factor, factor, n_factor], [0, n_factor, 0]]) # sharpen
	return cv2.filter2D(img, -1, kernel)

# Takes in an image and value [0,1]. Blurs the image according to the value (high value -> more blur)
def blur(img, value) :
	factor = math.floor(10 * value)
	kernel = np.ones((factor,factor),np.float32)/pow(factor, 2) # blur
	return cv2.filter2D(img, -1, kernel)

def edge_detect(img, value) :
	vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #  vertical gradient
	horizontal = np.array([[-1, -2, -2], [0, 0, 0], [1, 2, 2]]) # horizontal gradient
	if value == 0 or value == 1 or value == 2 or value == 6: # grey result
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

def updatefig(fig):
    global a
    if a == 0: 
    	im.set_array(dst)
    	im.set_cmap('Greys_r')
    	a = 1
    elif a == 1: 
    	im.set_array(v_grad)
    	im.set_cmap('Greys_r')
    	a = 2
    elif a == 2:
    	im.set_array(h_grad)
    	im.set_cmap('Greys_r')
    	a = 0
    return im,

# file = 'tree'
# file = 'clown'
# file = 'guitarist'
# file = 'fox'																																																																																																																																																																
# file = 'shape'
# file = 'chemise'
# file = 'city'
file = 'parasol'
img = cv2.imread('./images/' + file + '.jpg', 1)

dst = edge_detect(img, 6)
v_grad = edge_detect(img, 1)
h_grad = edge_detect(img, 2)

fig = plt.figure()
im = plt.imshow(dst, animated=True, cmap='Greys_r')
a = 0
ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)

# f = plt.figure(1, figsize=(20,5))
# ax1 = f.add_subplot(121)
# ax2 = f.add_subplot(122)
# ax1.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# ax2.imshow(cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB))
# ax1.imshow(dst, cmap='Greys_r')
# ax2.imshow(dst1, cmap='Greys_r')

plt.show()

