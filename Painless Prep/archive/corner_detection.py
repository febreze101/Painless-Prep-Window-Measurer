import cv2
import numpy as np

# Load the image
image = cv2.imread('imgs/close_1.jpg')  # Update the path to your image
resized_img = cv2.resize(image, (720, 1080))
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)


# # Increase contrast using histogram equalization
# gray = cv2.equalizeHist(gray)

cv2.imshow("gaussian ", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Initialize the QR code detector
qr_code_detector = cv2.QRCodeDetector()

# Detect and decode the QR codes
retval, decoded_info, points, _ = qr_code_detector.detectAndDecodeMulti(gray)
print(retval)
print(points)
print(decoded_info)
# Draw bounding boxes and show coordinates
if points is not None:
    for point in points:
        # Convert the coordinates to integer values
        points_int = point.astype(int)

        # Draw lines between the corner points
        for j in range(len(points_int)):
            # Draw lines between the corners
            cv2.line(resized_img, tuple(points_int[j]), tuple(points_int[(j + 1) % len(points_int)]), (0, 255, 0), 2)

            # # Annotate each point with its coordinates
            # coordinate_text = f"({points_int[j][0]}, {points_int[j][1]})"
            # cv2.putText(resized_img, coordinate_text, tuple(points_int[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the image with bounding boxes and coordinates
cv2.imshow("Image with QR Code Bounding Boxes and Coordinates", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
