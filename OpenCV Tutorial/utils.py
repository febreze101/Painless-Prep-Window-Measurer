import numpy as np
import cv2

def get_limits(color):
    """
    Get HSV color limits for detection

    Args:
        color: BGR color value  

    Returns:
        lowerLimit, upperLimit: HSV range for color detection
    """
    
    c = np.uint8([[color]]) # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    hue = hsvC[0][0][0]
    
    print(f"BGR color: {color}, HSV color: {hsvC[0][0]}")
    
    # Handle different color ranges
    if color[1] > color[0] and color[1] > color[2]:     # if green is dominant color
        # Green range in HSV space
        lowerLimit = np.array([75,89,20], dtype=np.uint8)
        upperLimit = np.array([95,240,240], dtype=np.uint8)
    elif hue >= 165:    # Red range
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0,  100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else: # other colors
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    print(f"Lower limit: {lowerLimit}")
    print(f"Upper limit: {upperLimit}")

    return lowerLimit, upperLimit

# Example
if __name__ == "__main__":
    red = [0, 0, 200]       # BGR format
    green = [0, 255, 0]     # BGR format
    
    print("Testing Red Detection: ")
    red_lower, red_upper = get_limits(red)
    
    print("Testing Green Detection: ")
    green_lower, green_upper = get_limits(green)
    