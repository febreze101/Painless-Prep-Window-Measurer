import os, cv2

cwd = os.getcwd()
img = cv2.imread(os.path.join(cwd, 'imgs', 'resized_pic.jpg'))

k_size = 7
img_blur = cv2.blur(img, (k_size, k_size))
img_blur_g = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_blur_m = cv2.medianBlur(img, k_size)

cv2.imshow('blur', img_blur)
cv2.imshow('blur_g', img_blur_g)
cv2.imshow('blur_m', img_blur_m)
cv2.waitKey(0)

cv2.destroyAllWindows()