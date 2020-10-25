import numpy as np
import cv2
img_size = 256
full_background = cv2.imread('background.jpg')
bg_height, bg_width, _ = full_background.shape
cut_size = np.random.randint(low=50, high=100)
start_cut_x = np.random.randint(low=0, high=bg_width-cut_size)
end_cut_x = start_cut_x + cut_size
start_cut_y = np.random.randint(low=0, high=bg_height-cut_size)
end_cut_y = start_cut_y + cut_size
img = full_background[start_cut_y: end_cut_y, start_cut_x: end_cut_x]
img = img/4 + 255//4*3
img = cv2.resize(img, (img_size, img_size))
center = [img_size//2 + np.random.randint(low=-20, high=20), 
  img_size//2 + np.random.randint(low=-20, high=20)]
n_points = np.random.randint(low=3, high=8)
condition = True
while condition:
  condition = False
  angles = np.random.rand(n_points)*np.pi*2
  angles = np.sort(angles)
  for i in range(n_points):
    d_angle = angles[(i+1)%n_points] - angles[i]
    if d_angle<0:
      d_angle+=np.pi*2
    if d_angle<np.pi/10:
      condition = True
      break
  
points = []
for angle in angles:
  if angle<np.pi/2:
    radius = min(img_size - center[0], img_size -  center[1])
    radius = np.random.randint(low=int(radius*0.8), high=radius)
  elif angle<np.pi:
    radius = min(center[0], img_size -  center[1])
    radius = np.random.randint(low=int(radius*0.8), high=radius)
  elif angle<1.5*np.pi:
    radius = min(center[0], center[1])
    radius = np.random.randint(low=int(radius*0.8), high=radius)
  else:
    radius = min(img_size - center[0], center[1])
    radius = np.random.randint(low=int(radius*0.8), high=radius)
  point = (int(center[0] + np.cos(angle)*radius), int(center[1] + np.sin(angle)*radius))
  points.append(point)
color = [np.random.randint(low=0, high=255), np.random.randint(low=0, high=255), np.random.randint(low=0, high=255)]
for i in range(n_points):
  point1 = points[i]
  point2 = points[(i+1)%n_points]
  cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]) , color, 3)
point = points[0]
cv2.circle(img, (center[0], center[1]), 10 , (255,0,0))
cv2.circle(img, (point[0], point[1]), 10 , (255,0,0))
cv2.imwrite('test.jpg', img)