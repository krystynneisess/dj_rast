# TODO:  Write accompanying README.

import numpy as np
import math as m
from attribute_data import AttributeData
from bpm_data import BPMData

class SongDataAPI:

	"""
	Initializer.  Initializes Attribute/BPM Data members for the provided song.

	Input:  song_name		The name of the song. 
							Eg. "fly_me_to_the_moon"
	"""
	def __init__(self, song_name = None):
		self.attributes = {
			"loudness_centroid": AttributeData(),
			"loudness_spread": AttributeData(),
			"frequency_centroid": AttributeData(),
			"frequency_spread": AttributeData(),
			"energy": AttributeData(),
			"bpm": BPMData()
		}

		self.member_getters = {
			"seconds": get_seconds,
			"file_name": get_file_name,
			"song_name": get_song_name,
			"song_length": get_song_length,
			"data": get_data,
			"data_type": get_data_type,
			"data_length": get_data_length,
			"values_per_point": get_values_per_point,
			"sensitivity": get_sensitivity,
			"bpm": get_bpm,
			"bpm_data": get_bpm_data
		}

		if song_name != None:
			a_lc = self.attributes["loudness_centroid"]
			a_ls = self.attributes["loudness_spread"]
			a_fc = self.attributes["frequency_centroid"]
			a_fs = self.attributes["frequency_spread"]
			a_e = self.attributes["energy"]
			a_bpm = self.attributes["bpm"]
			
			# populate data
			a_lc.populate_data(song_name + "_csv_files/" + song_name + ".wav.loudness_centroid.csv")
			a_ls.populate_data(song_name + "_csv_files/" + song_name + ".wav.loudness_spread.csv")
			a_fc.populate_data(song_name + "_csv_files/" + song_name + ".wav.frequency_centroid.csv")
			a_fs.populate_data(song_name + "_csv_files/" + song_name + ".wav.frequency_spread.csv")
			a_e.populate_data(song_name + "_csv_files/" + song_name + ".wav.energy.csv")
			a_bpm.populate_data(song_name + "_csv_files/" + song_name + ".wav.energy.csv")

			# normalize data
			a_lc.normalize()
			a_ls.normalize()
			a_fc.normalize()
			a_fs.normalize()
			a_e.normalize()
			# a_bpm.normalize()

	"""
	Convenience get-methods for each attribute's members.
	(See README for additional attribute/member descriptions.)

	Input:	attribute_name 	Attribute name. Can take on the values:
								"loudness_centroid"
								"loudness_spread"
								"frequency_centroid"
								"frequency_spread"
								"energy"
								"bpm"
			member_name 	Member name.  Can take on the values:
								"seconds"
								"file_name"
								"song_name"
								"song_length"
								"data"
								"data_type"
								"data_length"
								"values_per_point"
								"bpm" 			<"bpm" attribute only>
								"bpm_overall"   <"bpm" attribute only>
	"""
	def get_attribute(self, attribute_name):
		return self.attributes[attribute_name]
	
	def get_member(self, attribute_name, member_name):
		a = self.get_attribute(attribute_name)

		return self.member_getters[member_name](a)

	"""
	Downsamples the given attribute's data to the specified number of samples.
	Returns the downsampled data, but also updates the member variables 
	accordingly.

	Input:	attribute_name	Attribute name.
			num_samples  	The number of resultant samples.
								
	"""
	def downsample_data(self, attribute_name, num_samples):
		a = self.get_attribute(attribute_name)
		
		return a.downsample(num_samples)

"""
Module helper methods--you shouldn't need to call these yourself.
"""
def get_seconds(a):
	return a.seconds
def get_file_name(a):
	return a.file_name
def get_song_name(a):
	return a.song_name
def get_song_length(a):
	return a.song_length
def get_data(a):
	return a.data
def get_data_type(a):
	return a.data_type
def get_data_length(a):
	return a.data_length
def get_values_per_point(a):
	return a.values_per_point
def get_sensitivity(a):
	return a.sensitivity
def get_bpm(a):
	return a.bpm
def get_bpm_data(a):
	return a.bpm_data

# testing/playing around
if __name__ == "__main__":
	# s = SongDataAPI("fly_me_to_the_moon")
	# s = SongDataAPI("johnny_guitar")
	s = SongDataAPI("this_game")

	print s.get_member("loudness_centroid", "data_type")
	print s.get_member("loudness_spread", "data_type")
	print s.get_member("frequency_centroid", "data_type")
	print s.get_member("frequency_spread", "data_type")
	print s.get_member("energy", "data_type")
	print s.get_member("bpm", "data_type")