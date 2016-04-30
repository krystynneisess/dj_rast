++============================++
|| README for the AnimateAPI  ||
++============================++

------------------------------------
LOADING AN IMAGE
------------------------------------
To load an image in OpenCV or using the API, add it to the folder Images. Also change (I've just been adding as comments) the file variable to be the title of the image you want to use. For example, if the image you wanted was called "turtle.jpg", set file = 'turtle'. The actual loading is done with:
img = cv2.imread('./images/' + file + '.jpg', 1). 
This loads the image in as BGR (not RGB) into img. To change the image to a GrayScale change the 1 in the imread call to a 0. 
You can also change the image to various types with this command: 
	img = cv2.cvtColor(img, COLOR_BGR2***Somethingelse***)
The most useful of the COLOR options are: 
	cv2.COLOR_BGR2RGB
	cv2.COLOR_BGR2HSV (hue saturation value)

For convenience I change the image to RGB immediately

------------------------------------
DISPLAYING AN IMAGE
------------------------------------
To display an image I use matplotlib. It can be really handy to just see what an image looks on its own (after a filter, before a filter, etc) outside of the animation window. To open a figure call 
f = plt.figure(fignumber, figsize=(20,5)). (the animation window is fignumber = 1) 
You can then show the image with plt.imshow(img) or plt.imshow(img, cmap='Greys_r') if the image is in Grayscale. 
OR
You can display multiple images on a single figure!!!!!!!!!!!! :D D D D D D D D 
To do this run the following lines after making a figure 
ax1 = f.add_subplot(121) 
ax2 = f.add_subplot(122)
The notation here is MatLab notation, ax1 is the first block of the grid made by breaking f into a 1 by 2 block
Then to show an image on a subplot do 
ax1.imshow(img)
This is really useful for seeing the changes in any two images after applying a filter. 

------------------------------------
FILTERING AN IMAGE
------------------------------------
As of now I have the following functions to filter an image, they each return an np array corresponding to a new image, so original img is preserved. 
	sharpen(img, value)
	blur(img, value)
	edge_detect(img, value)

Sharpen and Blur both take in an image (either gray or color) and a value [0, 1] and behave as you'd expect. 
Edge_detect however takes in an image (either gray or color) and a value 0 - 6. I'm thinking about making this [0 1] as well. But for now here are the different edge detections each value gives:  
	0: Normal edge detection (gray scale)
	1: Vertical edge detection (gray scale)
	2: Horizontal edge detection (gray scale)
	3: Normal edge detection (color)
	4: Vertical edge detection (color)
	5: Horizontal edge detection (color)
	6: SUPER EDGE DETECT (gray scale refined)

------------------------------------
ANIMATING
------------------------------------
To animate open a figure 
	fig = plt.figure()
Then show a dummy image on the figure with some special flags:
	im = plt.imshow(dst, animated=True, cmap='Greys_r') (the cmap='Greys_r' is only necessary if displaying a gray image)
Next call the following command: 
	ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
Now the animation window is ready. The figure will be updated through the update fig function. Currently I have it set to just cycle based on a value of a. There are also some things you'd probably like to precompute especially for edge detection. The way I envision it working is that each call to update fig we iterate another step in the nparray representing the audio data. (or iterate several steps depending on the interval step).  
