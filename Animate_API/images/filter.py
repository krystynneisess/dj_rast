from saturation import *
from brightness import *
from temperature import *
from contrast import *
from invert import *


filters = {'saturate': saturate,
		   'desaturate': desaturate,
		   'brightness': brightness,
		   'warmer': warmer,
		   'cooler': cooler,
		   'contrast': contrast,
		   'invert': invert,
		   'sharpen': sharpen,
		   'blur': blur,
		   'edge_detect', edge_detect}


def filter(img, filter_name, value):
	if filter_name not in filters:
		error = filter_name + ' is not a valid filter'
		raise NameError(error)
	else:
		filters[filter_name](img, value)


