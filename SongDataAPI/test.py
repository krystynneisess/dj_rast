import numpy as np
import math as m
from song_data_api import SongDataAPI
from hist_match import Equalizer

song = SongDataAPI("fly_me_to_the_moon")

loudness_spread = song.get_member("loudness_spread", "data")

