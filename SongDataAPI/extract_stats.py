import sys

"""
Takes in a file name--this should be a shape_statistics csv file
(eg. yadda_yadda.wav.spectral_shape_statistics.csv).

Writes each stat--centroid, spread, skewness, and kurtosis--into
correspondingly named csv files 
(eg. yadda_yadda.wav.spectral_centroid.csv).
"""
def extract_stats(filename):
	filename = filename.split(".")[0] + "_csv_files/" + filename

	f_stats = open(filename, "r")

	filename_tokens = filename.split("shape_statistics")

	f_centroid_name = filename_tokens[0] + "centroid" + filename_tokens[1];
	f_spread_name = filename_tokens[0] + "spread" + filename_tokens[1];
	f_skewness_name = filename_tokens[0] + "skewness" + filename_tokens[1];
	f_kurtosis_name = filename_tokens[0] + "kurtosis" + filename_tokens[1];

	f_centroid = open(f_centroid_name, "w")
	f_spread = open(f_spread_name, "w")
	f_skewness = open(f_skewness_name, "w")
	f_kurtosis = open(f_kurtosis_name, "w")

	line = f_stats.readline()
	while (line != ""):
		if (line[0] == "%"):
			# write entire line to all files
			f_centroid.write(line)
			f_spread.write(line)
			f_skewness.write(line)
			f_kurtosis.write(line)			
		else:
			# write each value to its respective file
			line_tokens = line.split(",")
			f_centroid.write(line_tokens[0].strip() + "\n")
			f_spread.write(line_tokens[1].strip() + "\n")
			f_skewness.write(line_tokens[2].strip() + "\n")
			f_kurtosis.write(line_tokens[3].strip() + "\n")

		line = f_stats.readline()

	f_stats.close()
	f_centroid.close()
	f_spread.close()
	f_skewness.close()
	f_kurtosis.close()

# Just swap out the file names as desired.
# extract_stats("fly_me_to_the_moon.wav.frequency_shape_statistics.csv")
if __name__ == "__main__":
	arg = sys.argv[1]
	extract_stats(arg + ".wav.frequency_shape_statistics.csv")



