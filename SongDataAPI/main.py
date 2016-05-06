import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from song_data_api import SongDataAPI
from cv_filter import *
import os.path
from Tkinter import *
import tkSnack
from hist_match import Equalizer
from contrast import Contrast
from edge_detect import EdgeDetect


interval = {}
interval["fly_me_to_the_moon"] = 400
interval["chum"] = 650

images = ['tiger','birds', 'chemise', 'city', 'clown', 
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
for song_str in songs:
	print song_str
song_str = raw_input('Please Select a Song: ')
while not os.path.isfile(song_str + '.wav'):
	song_str = raw_input(song_str + ' is not one of the available songs, please select another: ')

modes = ['separate', 'composite']
print ("----==== Available Modes ====----")
print ("separate: different filters are displayed on different images")
print ("composite: many filters are displayed on the same image")
mode = raw_input("Please Select a Mode: ")
while mode not in modes:
	mode = raw_input(mode + " is not one of the available modes, please select another: ")


song = SongDataAPI(song_str)
frequency_spread = song.get_member("frequency_spread", "data")
frequency_centroid = song.get_member("frequency_centroid", "data")
frequency_centroid = frequency_centroid[0:6000]
frequency_centroid = frequency_centroid/np.max(frequency_centroid)
loudness_spread = song.get_member("loudness_spread", "data")
song_length = song.get_member("loudness_spread", "song_length")
song_samples = song.get_member("loudness_spread", "data_length")
img = cv2.imread('images/' + img + '.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_edge = EdgeDetect(img)
img_cont = Contrast(img)

dst = img_edge.detect_edge(3)
dst_1 = np.array(dst, dtype=np.float32)
img_1 = np.array(img, dtype=np.float32)
dst1 = np.array(((dst_1 + img_1)/2), dtype=np.uint8)
eq_dst1 = Equalizer(dst1)
eq_img = eq_dst1.equalize_color()


bpm_bold = img_edge.detect_edge(6)
bpm_faded = img_edge.detect_edge_faded(6, 0.5)

avg_loud = np.mean(loudness_spread)
diff = max(abs(avg_loud - np.amax(loudness_spread)), abs(avg_loud - np.amin(loudness_spread)))

avg = np.mean(frequency_spread)
a = 0
toggle = 0
fig = plt.figure(2, figsize=(20,8))
my_interval = interval[song_str]

bpm = 400
increment = float(song_samples)*float(bpm)/(float(song_length)*float(1000))

if mode == 'separate':
	def updatefig(fig):
	    global a, avg, diff, toggle
	    a += increment

	    if toggle == 0:
	    	toggle = 1
	    	im_bpm.set_array(bpm_bold)
	    	im_bpm.set_cmap("Greys_r")

	    else:
	    	toggle = 0
	    	im_bpm.set_array(bpm_faded)
	    	im_bpm.set_cmap("Greys_r")

	    value = frequency_centroid[a]
	    value_contrast = abs(avg_loud - loudness_spread[m.floor(a)])/diff
	    img_sat = saturate(img, value)
	    img_temp = temperature(img, value)
	    img_bright = brightness(img, value)
	    im_sat.set_array(img_sat)
	    im_temp.set_array(img_temp)
	    im_bright.set_array(img_bright)
	    im_contrast.set_array(contrast(img, 1.0+value_contrast, value_contrast*20))
	    im_blur.set_array(blur(img, value_contrast))
	    return im_sat, im_temp, im_bright, im_contrast, im_blur, im_bpm

	ax1 = fig.add_subplot(231)
	ax2 = fig.add_subplot(232)
	ax3 = fig.add_subplot(233)
	ax4 = fig.add_subplot(234)
	ax5 = fig.add_subplot(235)
	ax6 = fig.add_subplot(236)
	im_sat = ax1.imshow(img, animated=True)
	ax1.set_title('Saturation by Frequency')
	im_temp = ax2.imshow(img, animated=True)
	ax2.set_title('Temperature by Frequency')
	im_bright = ax3.imshow(img, animated=True)
	ax3.set_title('Brightness by Frequency')
	im_contrast = ax4.imshow(img, animated=True)
	ax4.set_title('Contrast by Loudness')
	im_blur = ax5.imshow(img, animated=True)
	ax5.set_title('Blur by Loudness')
	im_bpm = ax6.imshow(img, animated=True)
	ax6.set_title('Edge Detection by BPM')
	ani = animation.FuncAnimation(fig, updatefig, interval=my_interval, blit=True)

else:
	def updatefig(fig):
	    global a, avg, diff
	    a += increment
	    value = frequency_centroid[a]
	    value_contrast = abs(avg_loud - loudness_spread[m.floor(a)])/diff
	    dst = blur(contrast(temperature(saturate(img, value), value), 1.0+value_contrast, value_contrast*20), value_contrast)
	    im.set_array(dst)
	    return im,

	im = plt.imshow(img, animated=True)
	ani = animation.FuncAnimation(fig, updatefig, interval=my_interval, blit=True)

	root = Tk()
	tkSnack.initializeSnack(root)
	song_to_play = tkSnack.Sound()
	song_to_play.read(song_str +'.wav')
	print "PLAYING MUSIC"
	song_to_play.play()
	print "PLOTTING START"
plt.show()