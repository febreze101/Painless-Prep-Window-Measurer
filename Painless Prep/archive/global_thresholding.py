import os, cv2

cwd = os.getcwd()

img = cv2.imread(os.path.join(cwd, 'imgs', 'resized_pic.jpg'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
cv2.waitKey(2000)

thresh = cv2.blur(thresh, (5, 5))
ret, thresh = cv2.threshold(thresh, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()