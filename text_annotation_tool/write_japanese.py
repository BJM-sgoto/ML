'''
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import time

## Make canvas and set the color
img = np.zeros((200,400,3),np.uint8)
b,g,r,a = 0,255,0,0


## Use simsum.ttc to write Chinese.
fontpath = "./simsun.ttc" # <== 这里是宋体路径 
font = ImageFont.truetype(fontpath, 32)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((50, 80),  "端午节就要到了。。。", font = font, fill = (b, g, r, a))
img = np.array(img_pil)


## Display 
cv2.imshow("res", img);
cv2.waitKey();
cv2.destroyAllWindows()
#cv2.imwrite("res.png", img)
'''
import cv2
import numpy as np


cord = (0,80)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)
#text = "Hello world"   # just work
text = "内容理解团队"  # messy text in the image
img = np.zeros((200,400,3),np.uint8)
cv2.putText(img, text,
            cord,
            font,
            fontScale,
            fontColor)

# Display the image
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()