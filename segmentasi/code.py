import numpy as np
import cv2
import kumpulanKode as libs

import glob

images = sorted(glob.glob('tomat/*.JPG'))
kernel = np.ones((5,5), np.uint8)
i= 1
j = 1

csv = open("mytomat.csv", "w")
for tomat in images:
	img = cv2.imread(tomat)
	red = 0
	orange = 0
	green = 0
	
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)
	
	#img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
	#b, g, r = cv2.split()
	#b = libs.treshold(libs.substract(r, g))
	
	ret, b = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	
	dilate = cv2.dilate(b, kernel, iterations = 10)
	final = cv2.erode(dilate, kernel, iterations= 10)
	hasil = libs.subrgbgray(img, final)

	row, col, ch  = hasil.shape
	for x in range (0, row):
		for y in range (0, col):
			b, g, r = hasil[x, y]
			if(b & g & r):
				if(r-g >90):
					red = red+1
				elif(r-g>50):
					orange = orange+1
				else:
					green = green+1
			
	print('Hasil1/%d/%d_%d.jpg' %(i, i, j))
	name = 'Hasil1/%d/%d_%d.jpg' %(i, i, j)
	
	myTomat = '%s, %d, %d, %d' %(name, red, orange, green)
	myTomat = myTomat + '\n'
	#print(myTomat)
	csv.write(myTomat)
	
	
	#cv2.imwrite(name, hasil)
	if(j == 15) :
		i = i + 1
		j = 0
	j = j + 1
	#name = '%s' %(tomat)
	#cv2.imwrite(name, img)
	#print(name)
