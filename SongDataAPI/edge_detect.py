import cv2
import numpy as np
import math as m

class EdgeDetect: 

	def __init__(self, img_data = None):
		self.data = img_data
		self.v_g = np.empty(1)
		self.h_g = np.empty(1)
		self.e_d_g = np.empty(1)
		self.v_c = np.empty(1)
		self.h_c = np.empty(1)
		self.e_d_c = np.empty(1)
		self.e_d_pro = np.empty(1)
		self.is_gray = (len(self.data.shape) == 1)
		self.edge_detect()

	def edge_detect(self) :
		vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #  vertical gradient
		horizontal = np.array([[-1, -2, -2], [0, 0, 0], [1, 2, 2]]) # horizontal gradient

		img2 = self.data
		img = cv2.bilateralFilter(img2, 15, 80, 80)
		img1 = np.array(img, dtype=np.float64)
 
		v_grad = cv2.filter2D(img1, -1, vertical)
		h_grad = cv2.filter2D(img1, -1, horizontal)

		abs_v = np.absolute(v_grad)
		abs_h = np.absolute(h_grad)

		v_grad = np.array(abs_v, dtype=np.uint8)
		h_grad = np.array(abs_h, dtype=np.uint8)

		if not self.is_gray:
			img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			v_grad_g = cv2.filter2D(img_g, -1, vertical)
			h_grad_g = cv2.filter2D(img_g, -1, horizontal)

			abs_v_g = np.absolute(v_grad_g)
			abs_h_g = np.absolute(h_grad_g)

			v_grad_g = np.array(abs_v_g, dtype=np.uint8)
			h_grad_g = np.array(abs_h_g, dtype=np.uint8)

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
			self.v_c = v_grad
			self.h_c = h_grad
			self.v_g = v_grad_g
			self.h_g = h_grad_g
			self.e_d_c = dst
		else:
			self.v_g = v_grad
			sefl.h_g = h_grad
			v_grad_g = v_grad
			h_grad_g = h_grad
			img_g = img 

		height, width = img_g.shape
		dst1 = np.empty((height, width), dtype=np.uint8)
		#edge detection grayscale:
		for i in range(height):
			for j in range(width):
				sum0 = pow(v_grad_g[i, j], 2) + pow(h_grad_g[i, j], 2)
				dst1[i, j] = pow(sum0, .5)
		self.e_d_g = dst1
		kernel = np.ones((7,7),np.float32)/pow(7, 2)

		dst2 = cv2.filter2D(dst1, -1, kernel)

		dstv = cv2.filter2D(dst2, -1, vertical)
		dsth = cv2.filter2D(dst2, -1, horizontal)

		for i in range(height):
			for j in range(width):
				sum0 = pow(dstv[i, j], 2) + pow(dsth[i, j], 2)
				dst2[i, j] = pow(sum0, .5)
		self.e_d_pro = dst2

	def detect_v(self, color) :
		if color : 
			return self.v_c
		else: 
			return self.v_g

	def detect_h(self, color) : 
		if color : 
			return self.h_c
		else: 
			return self.h_g

	def detect_e(self, color) : 
		if color : 
			return self.e_d_c
		else: 
			return self.e_d_g

	def detect_edge(self, value) :
		if value == 0:
			return self.detect_e(False)
		elif value == 1:
			return self.detect_v(False)
		elif value == 2: 
			return self.detect_h(False)
		elif value == 3: 
			return self.detect_e(True)
		elif value == 4: 
			return self.detect_v(True)
		elif value == 5: 
			return self.detect_h(True)
		elif value == 6: 
			return self.e_d_pro
