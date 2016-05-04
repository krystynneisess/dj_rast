import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from song_data_api import SongDataAPI
from cv_filter import *
import os.path
from Tkinter import *
import tkSnack


interval = {}
interval["fly_me_to_the_moon"] = 400
interval["chum"] = 650

images = ['small','birds', 'chemise', 'city', 'clown', 
          'fox_1','fox_2', 'guitarist', 'parasol', 
          'seascape', 'shape', 'tree', 'under_expose']
print('----===== Available Images =====----')
for image in images:
	print image
img = raw_input('Please Select an Image: ')

while not os.path.isfile('images/' + img + '.jpg'):
	img = raw_input(img + ' is not one of the available images, please select another: ')

songs = ['chum', 'fly_me_to_the_moon']
print('----===== Available Songs =====----')
for song in songs:
	print song
song = raw_input('Please Select a Song: ')
while not os.path.isfile(song + '.wav'):
	song = raw_input(song + ' is not one of the available songs, please select another: ')

modes = ['separate', 'composite']
print ("----==== Available Modes ====----")
print ("separate: different filters are displayed on different images")
print ("composite: many filters are displayed on the same image")
mode = raw_input("Please Select a Mode: ")
while mode not in modes:
	mode = raw_input(mode + " is not one of the available modes, please select another: ")


song = SongDataAPI(song)
frequency_spread = song.get_member("frequency_spread", "data")
frequency_centroid = song.get_member("frequency_centroid", "data")
frequency_centroid = frequency_centroid[0:6000]
frequency_centroid = frequency_centroid/np.max(frequency_centroid)
img = cv2.imread('images/' + img + '.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



avg = np.mean(frequency_spread)
a = 0
fig = plt.figure(2, figsize=(20,8))
my_interval = interval["fly_me_to_the_moon"]

if mode == 'separate':
	def updatefig(fig):
	    global a, avg, diff
	    a += 1
	    value = frequency_centroid[a]
	    img_sat = saturate(img, value)
	    img_temp = temperature(img, value)
	    img_bright = brightness(img, value)
	    im_sat.set_array(img_sat)
	    im_temp.set_array(img_temp)
	    im_bright.set_array(img_bright)
	    return im_sat, im_temp, im_bright

	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	im_sat = ax1.imshow(img, animated=True)
	ax1.set_title('saturation')
	im_temp = ax2.imshow(img, animated=True)
	ax2.set_title('temperature')
	im_bright = ax3.imshow(img, animated=True)
	ax3.set_title('brightness')
	ani = animation.FuncAnimation(fig, updatefig, interval=my_interval, blit=True)

else:
	def updatefig(fig):
	    global a, avg, diff
	    a += 1
	    value = frequency_centroid[a]
	    dst = saturate(img, value)
	    im.set_array(dst)
	    return im,

	im = plt.imshow(img, animated=True)
	ani = animation.FuncAnimation(fig, updatefig, interval=my_interval, blit=True)

# diff = max(abs(np.mean(loudness) - min(loudness)), abs(np.mean(loudness) - max(loudness))
# blur((loudness[x] - np.mean(loudness))/diff, img))

# con_img = Contrast(img)
# con = contrast(1.0 + loudness[x] - np.mean(loudness))/diff, 20*loudness[x])
# root = Tk()
# tkSnack.initializeSnack(root)
# song_to_play = tkSnack.Sound()
# song_to_play.read(song +'.wav')
# song_to_play.play()
plt.show()