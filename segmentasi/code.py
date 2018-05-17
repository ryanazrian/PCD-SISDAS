import numpy as np
import cv2
import kumpulanKode as libs

import glob

images = sorted(glob.glob('tomat/*.JPG'))
kernel = np.ones((5,5), np.uint8)
i= 1
for tomat in images:
	img = cv2.imread(tomat)
	
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)
	
	#img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
	#b, g, r = cv2.split()
	#b = libs.treshold(libs.substract(r, g))
	
	ret, b = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	
	dilate = cv2.dilate(b, kernel, iterations = 10)
	final = cv2.erode(dilate, kernel, iterations= 10)
	
	hasil = libs.subrgbgray(img, final)
	
	print('Hasil/%d.jpg' %(i))
	name = 'Hasil/%d.jpg' %(i)
	cv2.imwrite(name, hasil)
	i = i + 1
	#name = '%s' %(tomat)
	#cv2.imwrite(name, img)
	#print(name)
