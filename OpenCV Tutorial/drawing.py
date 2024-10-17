import os, cv2

cwd = os.getcwd()

img = cv2.imread(os.path.join(cwd, 'imgs', 'resized_pic.jpg'))

# line

# rectange

# circle

# text


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()