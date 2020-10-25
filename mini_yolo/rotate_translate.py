import cv2
import os
import numpy as np

background_folder = './background/'
backgrounds = [np.float32(cv2.imread(background_folder + x)) for x in os.listdir(background_folder)]

img = np.float32(cv2.imread('./full_dataset/train_image/20191014131023.jpg'))
rectangle = [267.34394233240954,162.7028260354743,101.40838852283275,47.49497667771576,5.327588303030494]
background = backgrounds[np.random.randint(low=0, high=len(backgrounds))]
h, w, _ = background.shape
start_cut_x = np.random.randint(low=0, high=w-640+1)
end_cut_x = start_cut_x + 640
start_cut_y = np.random.randint(low=0, high=h-480+1)
end_cut_y = start_cut_y + 480
background = background[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
background = background.copy()

rotate_angle = np.random.randint(low=0, high=360)
M = cv2.getRotationMatrix2D((rectangle[0], rectangle[1]),rotate_angle,1)
dx, dy = np.random.randint(low=-7, high=8, size=[2])
M += np.float32([[0, 0, dx],[0, 0, dy]])
img = img + 1
img = cv2.warpAffine(img, M, (640, 480))
mask = np.where(img>0.5, 1, 0)
img = (img - 1) * mask + background * (1 - mask)
angle = rectangle[4] - rotate_angle / 180 * np.pi
if angle < 0:
	angle += 2*np.pi
print('img.shape',img.shape)
cv2.imwrite('test.jpg', img)