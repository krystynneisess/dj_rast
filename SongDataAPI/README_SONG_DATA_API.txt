
++============================++
|| README for the SongDataAPI ||
++============================++

*** Note ***
You *probably* don't want to read through all this at once--which is quite alright!  It's really more of a reference manual, just here to be consulted as needed.
************





CONTENTS
--------------------------------
(0) Introduction

(1)	SongDataAPI Methods

(2) Attribute Glossary

(3) Member Glossary





(0) INTRODUCTION
--------------------------------
The SongDataAPI allows you to interface with a variety of audio features.  You should have a separate instance of the SongDataAPI class per song you wish to get data from.  

This class gives you access to the following audio attributes:
	- Loudness centroid
	- Loudness spread
	- Frequency centroid
	- Frequency spread
	- Energy
	- BPM

For each of which, there exists a host of member variables:
	- Seconds
	- File name
	- Song name
	- Song length
	- Data
	- Data type
	- Data length
	- Values per point
	- Sensitivity*
	- BPM*
	- BPM Data*
where those marked with an asterisk are available on the BPM attribute only.

Realistically, you probably won't be calling very many of these attributes.  At first, you'll primarily be interacting with only the Data, BPM, or BPM Data.  Nevertheless, they are available in case you'd like to play around or do some fancier tricks.  

See the "Attribute Glossary" or "Member Glossary" for more detailed descriptions of each attribute or member variable, respectively.





(1) SongDataAPI METHODS
--------------------------------
See the in-line comments if you seek more detailed usage instructions.  But if you'd like high-level fluff (and the occasional pointer), read on!

def __init__(self, song_name = None):
	The initalizer--just pass in the song name and all the SoundData/BPMData is automagically populated!  (See the "Member Glossary" for song_name semantics.)	

def get_attribute(self, attribute_name):
	Name the attribute you want, and you'll get it's corresponding SoundData/BPMData instance!

def get_member(self, attribute_name, member_name):
	Name the attribute and member you want, and you'll get the corresponding value.  (Which may be numpy array.  Or a float.  Or an int.  Or a string.  Or... uh, actually I think that's it.) 

def downsample_data(self, attribute_name, num_samples):
	So you want to downsample a particular attribute's data (i.e. numpy array "data" member).

	Name the attribute, and specify how many samples you want to end up with.  Hint!  If you'd like n samples per second, set num_samples to n * the attribute's song_length.  By default, there are ~40 samples/second.  

	A couple things to note:
		- Please don't try to trick it into "upsampling" by asking for more.  It will break.
		- Please don't repeatedly downsample the same data.  The implementation (at least currently) is a naiive approximation, and so error will rapidly accumulate.
		- The more you downsample by, the "better" (i.e. more accurate) the downsampling is.

		- TL;DR:  This should probably be made more robust... but, uh, I guess this'll do for now.





(2) ATTRIBUTE GLOSSARY
--------------------------------
The Data member of the following attributes is sampled data over time, unless otherwise specified.

"loudness_centroid":
	The centroid of a song's loudness coefficients.

	Derived from Yaafe's "PerceptualSharpness" feature.
	[http://yaafe.sourceforge.net/features.html#perceptualsharpness]

"loudness_spread:
	The standard deviation of a song's loudness coefficients.

	Derived form Yaafe's "PerceptualSpread" feature.
	[http://yaafe.sourceforge.net/features.html#perceptualspread]

"frequency_centroid":
	The centroid of a song's frequency envelope.

	Derived from Yaafe's "SpectralShapeStatistics" feature.
	[http://yaafe.sourceforge.net/features.html#spectralshapestatistics]
	
"frequency_spread":
	The standard deviation of a song's frequency envelope.

	Derived from Yaafe's "SpectralShapeStatistics" feature.
	[http://yaafe.sourceforge.net/features.html#spectralshapestatistics]

"energy":
	The energy of a song, as computed via root-mean-square.

	Derived from yaafe's "Energy" feature.
	[http://yaafe.sourceforge.net/features.html#energy]

"bpm":
	The estimated BPM of a song.  Data is available as a single value (i.e. "bpm" member, the estimated BPM for the entire song), or as samples over time (i.e. "bpm_data" member, the estimated BPM over brief intervals).

	Derived from the "energy" attribute.





(3) MEMBER GLOSSARY
--------------------------------
Each entry is formatted as follows:
<Member name>:
	<Python type>.  <Example value>
	<Additional description>

"seconds":
	Dictionary<String, Integer>.  {"fly_me_to_the_moon:163"}
	A class-level dictionary with mappings from song names to the song's duration in seconds.
	 
"file_name":
	String.  "fly_me_to_the_moon.wav.magical_energy.csv"
	The name of the CSV file from the SoundData/BPM Data instance was created.

"song_name":
	String.  "fly_me_to_the_moon"
	The name of the song.  Note that this is considered to be whatever text precedes the ".wav" in the name of the song's .wav file. 

"song_length":
	Integer.  "163"
	The length of the song in seconds.

"data":
	Numpy array of floats.  [[1.0, 2.0, 3.0][4.0, 5.0, 6.0]]
	An N x M array of sampled data, where N is the length of the data (see "data_length") and M is the number of values per sample point (see "values_per_point").

	However, typically this will be just an N x 1 Numpy array.

"data_type":
	String.  "magical_energy"
	The data's attribute type.  Other (perhaps more familiar) example values include "loudness_centroid", "frequency_spread".

"data_length":
	Integer.  2
	The number of samples stored in "data".  Typically, this will be in the hundreds or thousands.

"values_per_point":
	Integer.  3
	The number of values stored in each sample.  Typically, this will just be 1.

"sensitivity":
	Float.  1.15
	BPM attribute exclusive!  A factor describing the sensitivity threshold for beat detection.  

	That is, a sample is considered a "beat" if its instantaneous energy is greater than the average energy over adjacent samples by at least this factor.

"bpm":
	Float.  118.314159
	BPM attribute exclusive!  The average BPM of the entire song.

"bpm_data":
	Numpy array of floats. [[1][0]] 
	BPM attribute exclusive!  An N x 1 array, where N is the length of the data (see "data_length").  A value of 1 indicates a beat; a value of 0 indicates otherwise.
