
import numpy as np
import math as m
from attribute_data import AttributeData

class BPMData(AttributeData):
	"""
	Constructor.  

	Adds a field called sensitivity, a float describing the 
	threshold for detecting beats.
	"""
	def __init__(self):
		super(BPMData, self).__init__()

		self.sensitivity = 0.0
		self.bpm = 0
		self.bpm_data = np.zeros(0)

	"""
	Averages the nearest n values centered around the ith value of data.
	Returns np.zeros() if there are not enough points in vicinity to average over.

	n is assumed to be an odd integer.
	"""
	def avg_nearest(self, i, n):
		# return np.zeros() if there aren't enough values to average over
		if (i < (n / 2) or i > (len(self.data) - 1 - n / 2)):
			return np.zeros(self.values_per_point)

		# else, compute the running average
		avg_val = np.zeros(self.values_per_point)
		j = i - n / 2
		while (j <= i + n / 2):
			avg_val += (self.data[j] / float(n))
			j += 1

		return avg_val

	"""
	At intervals of step_size apart, averages a quantity of 
	block_size values centered around each step.  Performed over the 
	given numpy array, returning resuls as another numpy array.
	"""
	def avg_array(self, step_size, block_size):
		avg_data = []
		
		j = 0
		while j < len(self.data):
			avg_val = self.avg_nearest(j, block_size)
			avg_data.append(avg_val)
			j += step_size

		# set fields 
		# print "***" + str(avg_data)
		# self.data = np.array(avg_data)

		# return self.data
		return np.array(avg_data)

	"""
	Determines sensitivity based on variance in energy data. 

	Higher variance --> more sensitive --> lower sensitivity threshold.
	"""
	def set_sensitivity(self, filename):
		# TODO
		sens = {}
		sens["fly_me_to_the_moon"] = 1.170
		sens["chum"] = 1.0
		sens["johnny_guitar"] = 1.225
		sens["this_game"] = 1.080
	
		return sens[filename]

	"""
	Returns the variance in energy data.
	"""
	def energy_variance(self):
		# compute average
		avg = 0

		for d in self.data:
			avg += d[0]

		avg = avg / self.data_length

		# compute variance
		var = 0

		for d in self.data:
			var += m.pow(avg - d[0], 2)

		var = var / self.data_length

		return var

	"""
	From self.data (which is currently instantaneous energy measurements),
	extracts beats, where 1 = beat; 0 = no beat.
	"""
	def extract_beats(self):
		instant_e = self.data
		step_size = 1
		block_size = 43
		average_e = self.avg_array(step_size, block_size)

		beats = []

		j = 0
		while j < len(self.data):
			if np.linalg.norm(average_e[j]) == 0.0:
				beats.append(0)
			else:
				ratio = instant_e[j] / average_e[j]
				if ratio > self.sensitivity:
					beats.append(1)
				else:
					beats.append(0)
			j += step_size

		return np.array(beats)

	"""
	BPM calculation--attempt number one...
	"""
	def get_bpm_1(self):
		beats = self.extract_beats()
		# print beats
		beat_count = 0

		for b in beats:
			print b
			if b == 1:
				beat_count += 1

		bpm = (beat_count * 60.0) / self.song_length

		return bpm

		
	"""
	BPM calculation--attempt number two...
	"""
	def get_bpm_2(self):
		beats = self.extract_beats()

		# min_beat_len = 3 # NVM... # ideal for fly_me_to_the_moon
		min_beat_len = 2 # ideal for earl_sweatshirt_chum

		beat_count = 0
		curr_beat_len = 0

		for idx, b in enumerate(beats):
			if idx != 0:
				if b == 1:
					curr_beat_len += 1
				if beats[idx] < beats[idx - 1]: # downstep
					if curr_beat_len >= min_beat_len:
						beat_count += 1
					curr_beat_len = 0

		bpm = (beat_count * 60.0) / self.song_length

		return bpm

	"""
	Populates the BMPData object's fields from a given file.

	"""
	def populate_data(self, filename):
		super(BPMData, self).populate_data(filename)


		self.data_type = "bpm"
		self.sensitivity = self.set_sensitivity(filename)
		self.bpm = self.get_bpm_2()
		self.bpm_data = self.extract_beats()


if __name__ == "__main__":
	# Miscellaneous testing!
	a = BPMData()
	a.populate_data("fly_me_to_the_moon_csv_files/fly_me_to_the_moon.wav.energy.csv")
	# a.populate_data("chum_csv_files/chum.wav.energy.csv")
	# a.populate_data("johnny_guitar_csv_files/johnny_guitar.wav.energy.csv")
	# a.populate_data("this_game_csv_files/this_game.wav.energy.csv")

	# a.populate_data("dev_csv_files/fly_me_to_the_moon.wav.energy_b1024s1024.csv")
	# a.populate_data("dev_csv_files/fly_me_to_the_moon.wav.perceptual_sharpness.csv")
	# a.populate_data("dev_csv_files/toy_data_2.wav.toy_type.csv")
	print a.file_name
	print a.song_name
	print a.song_length
	# print a.data
	print a.data_type
	print a.data_length
	print a.values_per_point

	# avg = a.avg_array(1, 3)
	# print avg
	
	# print a.extract_beats()
	print a.bpm
	print a.energy_variance()

	# a.normalize()
	# print a.data

	# a.derivative()
	# print a.data
	# print a.data_length
	# print a.data_type

	# a.downsample(a.song_length)
	# print a.data



	#BPM REFERENCE vals

	# BPM ~BPM 		sens 	Var 		Name
	# 118: 118.9 	1.170	0.00627		fly_me_to_the_moon
	# 154: 155.7 	1.000	0.01288		chum 					
	# 080: 080.3 	1.225   0.00134 	johnny_guitar 			
	# 147: 146.3	1.080	0.00232		this_game 				