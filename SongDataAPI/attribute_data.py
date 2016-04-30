
import numpy as np 

class AttributeData(object):
	seconds = {}
	seconds["fly_me_to_the_moon"] = 163
	seconds["chum"] = 190
	# toy data for testing
	seconds["toy_data_1"] = 4
	seconds["toy_data_2"] = 4
	seconds["toy_data_3"] = 4
	seconds["toy_data_4"] = 4

	def __init__(self):
		# self.data_name = ""
		# self.song_name = "";
		self.file_name = ""

		self.song_name = ""
		self.song_length = 0

		self.data = []
		self.data_type = ""
		self.data_length = 0
		self.values_per_point = 0

	"""
	Converts a csv data file named self.filename into a 2D numpy array.

	For example, a csv 
		% A comment
		1.0, 2.0, 3.0
		4.0, 5.0, 6.0
	becomes perceptual_sharpness
		[[1.0, 2.0, 3.0][4.0, 5.0, 6.0]]

	Note that it does NOT become
		[["1.0, 2.0, 3.0"]["4.0, 5.0, 6.0"]]
	or 
		[["1.0", "2.0", "3.0"]["4.0", "5.0", "6.0"]]

	"""
	def file_to_nparray(self):
		f = open(self.file_name, "r")
		data = []

		line = f.readline()
		while (line != ""):
			# Don't store comments
			if (line[0] == "%"):
				line = f.readline()
				continue

			line_tokens = line.split(",")

			data_point = []
			for token in  line_tokens:
				data_point.append(float(token))

			data.append(data_point)

			line = f.readline()

		f.close()

		return np.array(data)

	"""
	A numerical approximation to downsampling.  Assumes that
	the original number of samples >> the new number of samples.

	Returns and sets self.data to the downsampled data as a numpy array.  
	"""
	def downsample(self, new_sample_num):
		N = self.data_length
		S = new_sample_num

		rr = N / S # reduction ratio--assumes integer division
				   # ie. how many samples we average over per new sample
		nc = 1.0 / rr # normalization constant

		new_data = []

		# averaging to appoximate downsampling
		for i in range(S):
			new_sample = np.zeros(self.values_per_point)
			for j in range(rr):
				new_sample += (nc * self.data[rr * i + j])

			new_data.append(new_sample)

		# update fields
		self.data = np.array(new_data)
		self.data_length = len(self.data)

		return self.data

	"""
	Returns the first-derivative of the sampled data.  It is
	recommended that this, if performed, be done prior to downsampling.
	"""
	def derivative(self):
		new_data = []

		for i in range(self.data_length - 1):
			new_point = self.data[i + 1] - self.data[i] # implicitly divide by "1 unit"
			new_data.append(new_point)

		new_data.append(new_data[-1]) # duplicate last value to keep same # of points

		# update fields
		self.data = np.array(new_data)
		self.data_type += "_deriv"

	"""
	Normalizes data into a [-1, 1] interval.  Returns and sets fields
	appropriately.
	"""
	def normalize(self):
		# determine largest magnitude value to normalize over
		max_magnitude = np.zeros(self.values_per_point)

		for point in self.data:
			for i in range(self.values_per_point):
				if abs(point[i]) > max_magnitude[i]:
					max_magnitude[i] = abs(point[i])

		# normalize!
		nc = np.array([1.0 / x for x in max_magnitude])
		norm_data = np.multiply(nc, self.data)

		# set fields
		self.data = norm_data

		return self.data

	"""
	Populates the AttributeData object's fields from a given file.
	"""
	def populate_data(self, filename):
		# filename = "fly_me_to_the_moon.wav.perceptual_sharpness.csv"
		self.file_name = filename

		filename_tokens = filename.split("/")
		filename_tokens = filename_tokens[-1].split(".")		
		# filename_tokens = filename.split(".")

		self.song_name = filename_tokens[0]	
		self.song_length = self.seconds[self.song_name]		
		
		self.data = self.file_to_nparray() 			 	
		self.data_type = filename_tokens[2] 		
		self.data_length = len(self.data)
		self.values_per_point = len(self.data[0])



if __name__ == "__main__":
	# Miscellaneous testing!
	a = AttributeData()
	# a.populate_data("dev_csv_files/fly_me_to_the_moon.wav.perceptual_sharpness.csv")
	a.populate_data("dev_csv_files/toy_data_4.wav.toy_type.csv")
	print a.file_name
	print a.song_name
	print a.song_length
	print a.data
	print a.data_type
	print a.data_length
	print a.values_per_point

	a.normalize()
	print a.data

	# print a.data.shape

	# for d in a.data:
	# 	print d

	# a.derivative()
	# print a.data
	# print a.data_length
	# print a.data_type

	# a.downsample(a.song_length)
	# print a.data

