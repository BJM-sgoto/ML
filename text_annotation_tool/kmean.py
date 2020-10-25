import cv2
import numpy as np
img = np.float32(cv2.imread('test.jpg'))
h, w, _ = img.shape
c1 = np.float32(np.zeros(3))
c2 = np.float32(np.ones(3) * 255)
print(c1, c2)
for i in range(5):
	d1 = np.sqrt(np.sum(np.power(img - c1, 2), axis=2))
	d2 = np.sqrt(np.sum(np.power(img - c2, 2), axis=2))
	g1 = np.expand_dims(np.float32(np.less(d1, d2)), axis=2)
	g2 = 1.0 - g1
	c1 = np.sum(img*g1,axis=(0,1))/ np.sum(g1)
	c2 = np.sum(img*g2,axis=(0,1))/ np.sum(g2)
	print(c1, c2)
cv2.imwrite("output.png", g1*255)