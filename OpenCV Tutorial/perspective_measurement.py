import cv2
import numpy as np
from scipy.spatial.distance import euclidean

def order_points(pts):
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
    
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have smallest sum, bottom right will have largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have smallest difference, bottom-left will have largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_perspective_transform(image, pts):
    """Get bird's eye view of image"""
    input_rect = order_points(pts)
    (t_left, t_right, b_right, b_left) = input_rect
    
    # Compute width of new image
    widthA = euclidean(b_right, b_left)
    widthB = euclidean(t_right, t_left)
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height of new image
    heightA = euclidean(t_left, b_left)
    heightB = euclidean(t_right, b_right)
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct destination points for transform
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')
    
    # calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(input_rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped, M, maxWidth, maxHeight


def get_object_ddimensions(image, ref_object_points, ref_width, ref_heihgt):
    """Get real-world dimensions using reference object"""
    # Get bird's eye view
    warped, M, maxWidth, maxHeight = get_perspective_transform(image, ref_object_points)
    
    # Calculate pixels per metric
    pixels_per_metric_w = maxWidth / ref_width
    pixels_per_metric_h = maxHeight / ref_heihgt
    
    return warped, pixels_per_metric_w, pixels_per_metric_h

def measure_object(image, object_points, pixel_per_metric_w, pixels_per_metric_h):
    """Measure object using calculated pixels per metric"""
    rect = order_points(object_points)
    (t_left, t_right, b_right, b_left) = rect
    
    # Calculate width and height in pixels
    widthA = euclidean(b_right, b_left)
    widthB = euclidean(t_right, t_left)
    heightA = euclidean(t_left, b_left)
    heightB = euclidean(t_right, b_right)
    
    # Take average of two sidies
    avg_width_pixels = np.mean(widthA, widthB)
    ave_height_pixels = np.mean(heightA, heightB)
    
    # Convert to real-world units
    real_width = avg_width_pixels / pixel_per_metric_w
    real_height = ave_height_pixels / pixels_per_metric_h
    
    return real_width, real_height