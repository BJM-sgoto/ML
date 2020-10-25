import numpy as np 
import cv2
img = np.zeros([128, 128])
npts = []
r = 64
for i in range(30):
	angle = (12 * i + 6)/180*np.pi
	x = 64 + r * np.cos(angle)
	y = 64 + r * np.sin(angle)
	npts.append([x, y])
npts = np.int32([npts])
cv2.fillPoly(img, npts, 255)
cv2.imwrite('test.jpg', img)