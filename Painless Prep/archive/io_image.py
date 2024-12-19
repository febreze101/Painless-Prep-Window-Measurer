import os
import cv2

cwd = os.getcwd()

# read image
image_path = os.path.join(cwd, 'imgs', '_FBF9200.jpg')

img = cv2.imread(image_path)

# write image
cv2.imwrite(os.path.join(cwd, 'imgs', 'pic.jpg'), img)

# visualize img
cv2.imshow('img', img)
cv2.waitKey(0)