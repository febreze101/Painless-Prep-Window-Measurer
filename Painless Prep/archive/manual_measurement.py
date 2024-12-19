import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def measure_object(image, object_points, pixels_per_metric_w, pixels_per_metric_h):
    """
    Measure object dimensions using calculated pixels per metric ratios
    Returns width and height in the same units as reference object (e.g., cm)
    """
    # Order points consistently
    rect = order_points(object_points)
    (tl, tr, br, bl) = rect
    
    # Calculate width (average of top and bottom)
    width_pixels_top = euclidean(tr, tl)
    width_pixels_bottom = euclidean(br, bl)
    avg_width_pixels = (width_pixels_top + width_pixels_bottom) / 2
    
    # Calculate height (average of left and right)
    height_pixels_left = euclidean(tl, bl)
    height_pixels_right = euclidean(tr, br)
    avg_height_pixels = (height_pixels_left + height_pixels_right) / 2
    
    # Convert to real-world units using pixels per metric ratios
    real_width = avg_width_pixels / pixels_per_metric_w
    real_height = avg_height_pixels / pixels_per_metric_h
    
    return real_width, real_height

def click_event(event, x, y, flags, params):
    """Handle mouse clicks to collect corner points"""
    img, points = params
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', img)

def test_measurement():
    # Load an image
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image")
        return
    
    # Create copies for drawing
    image_copy = image.copy()
    
    # Lists to store points
    ref_points = []
    object_points = []
    
    print("First, click the 4 corners of your reference object (e.g., credit card)")
    print("Click in this order: top-left, top-right, bottom-right, bottom-left")
    
    # Collect reference object points
    cv2.imshow('Image', image_copy)
    cv2.setMouseCallback('Image', click_event, (image_copy, ref_points))
    cv2.waitKey(0)
    
    if len(ref_points) != 4:
        print("Need exactly 4 points for reference object")
        return
    
    # Known dimensions of reference object (e.g., credit card in cm)
    REF_WIDTH = 8.56   # credit card width in cm
    REF_HEIGHT = 5.398 # credit card height in cm
    
    # Calculate pixels per metric using reference object
    ref_points = np.array(ref_points, dtype="float32")
    ref_points = order_points(ref_points)
    (tl, tr, br, bl) = ref_points
    
    # Calculate reference object width and height in pixels
    ref_width_pixels = euclidean(br, bl)
    ref_height_pixels = euclidean(tr, br)
    
    # Calculate pixels per metric
    pixels_per_metric_w = ref_width_pixels / REF_WIDTH
    pixels_per_metric_h = ref_height_pixels / REF_HEIGHT
    
    print("\nNow, click the 4 corners of the object you want to measure")
    print("Click in this order: top-left, top-right, bottom-right, bottom-left")
    
    # Collect object points
    cv2.setMouseCallback('Image', click_event, (image_copy, object_points))
    cv2.waitKey(0)
    
    if len(object_points) != 4:
        print("Need exactly 4 points for object")
        return
    
    # Convert points to numpy array
    object_points = np.array(object_points, dtype="float32")
    
    # Measure object
    width, height = measure_object(image, object_points, pixels_per_metric_w, pixels_per_metric_h)
    
    # Draw results
    cv2.polylines(image_copy, [ref_points.astype(int)], True, (0, 255, 0), 2)
    cv2.polylines(image_copy, [object_points.astype(int)], True, (0, 0, 255), 2)
    
    # Display measurements
    print(f"\nMeasurements:")
    print(f"Width: {width:.2f} cm")
    print(f"Height: {height:.2f} cm")
    
    # Show final image
    cv2.putText(image_copy, f"Width: {width:.2f}cm", 
                tuple(object_points[0].astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(image_copy, f"Height: {height:.2f}cm", 
                tuple(object_points[1].astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    
    cv2.imshow('Measurements', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the test
test_measurement()