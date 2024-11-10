import cv2
import numpy as np
from utils import get_limits
from PIL import Image

# Known dimensions of reference object (e.g., credit card in cm)
REF_WIDTH = 8.56  # credit card width in cm
REF_HEIGHT = 5.398  # credit card height in cm

def process_frame(frame, u_left_color, b_right_color, min_area=3500, kernel_size=3):
    """
    Process a frame to detect color with noise reduction
    
    Args:
        frame: Input frame
        color: BGR color to detect
        min_area: Minimum area threshold for detection
        kernel_size: Size of kernel for morphological operations
    """
    frame = cv2.flip(frame, 1)  # 1 for horizontal flip, 0 for vertical, -1 for both
    def detect_color(color, color_name):
        
        # Convert to HSV
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color limits
        lowerLimit, upperLimit = get_limits(color=color)
        
        # Print color limits for debugging
        # print(f"{color_name} HSV limits - Lower: {lowerLimit}, Upper: {upperLimit}")
        
        # Create mask
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        
        # Morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=2)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=3)
        opened_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernel)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        
        return {
            'original_mask': mask,
            'eroded_mask': eroded_mask,
            'dilated_mask': dilated_mask,
            'opened_mask': opened_mask,
            'closed_mask': closed_mask
        } 
    
    # Detect both colors
    red_masks = detect_color(u_left_color, "Red")
    green_masks = detect_color(b_right_color, "Green") 
    
    # Find contours for both colors
    u_left_contours, _ = cv2.findContours(red_masks['closed_mask'], 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
    b_right_contours, _ = cv2.findContours(green_masks['closed_mask'], 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
    
    ref_objects = []
    
    # Process red contours
    for contour in u_left_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            ref_objects.append({
                'position': 'upper-left',
                'coords': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area
            })
            cv2.putText(frame, f'Red: {int(area)}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Process green contours
    for contour in b_right_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ref_objects.append({
                'position': 'bottom-right',
                'coords': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area
            })
            cv2.putText(frame, f'Green: {int(area)}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if len(ref_objects) == 2:
        # get pixel per metric
        pixels_per_metric_w, pixels_per_metric_h = get_pixel_per_metric(ref_objects=ref_objects)
        
        # sort by y-coordinate to determine which is top and bottom
        ref_objects.sort(key=lambda x: x['center'][1])
        top_ref = ref_objects[0]
        bottom_ref = ref_objects[1]
        
        # Draw window pane boundaries
        window_coords = {
            'top_left': (top_ref['coords'][0], top_ref['coords'][1]),
            'bottom_right': (bottom_ref['coords'][0] + bottom_ref['coords'][2],
                             bottom_ref['coords'][1] + bottom_ref['coords'][3])
        }
        
        # Draw window outline
        cv2.rectangle(frame,
                      window_coords['top_left'],
                      window_coords['bottom_right'],
                      (255, 0, 0), 1)
        
        # Calculate window dimensions in pixels
        window_width_in_px = window_coords['bottom_right'][0] - window_coords['top_left'][0]
        window_height_in_px = window_coords['bottom_right'][1] - window_coords['top_left'][1]
        
        # Display dimensions (pixels)
        # cv2.putText(frame, f'Width: {window_width_in_px}px',
        #             (window_coords['top_left'][0], window_coords['top_left'][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # cv2.putText(frame, f'Height: {window_height_in_px}px',
        #             (window_coords['top_left'][0], window_coords['top_left'][1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        
        # convert window dimensions to metric
        window_height_in_metric = window_height_in_px * pixels_per_metric_h
        window_width_in_metric = window_width_in_px * pixels_per_metric_w
        
        # Display dimensions (metric)
        cv2.putText(
            frame,
            f'Width: {window_width_in_metric/100:.1f} cm',
            (window_coords['top_left'][0], window_coords['top_left'][1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f'Height: {window_height_in_metric/100:.1f} cm',
            (window_coords['top_left'][0], window_coords['top_left'][1] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
    
    return frame, red_masks, green_masks, ref_objects

    
def get_pixel_per_metric(ref_objects):
    top_ref = ref_objects[0]
    bottom_ref = ref_objects[1]
    
    top_ref_width_px = top_ref['coords'][2]
    top_ref_height_px = top_ref['coords'][3]
    bottom_ref_width_px = bottom_ref['coords'][2]
    bottom_ref_height_px = bottom_ref['coords'][3]
    
    pixels_per_metric_w = (top_ref_width_px + bottom_ref_width_px) / 2 / REF_WIDTH
    pixels_per_metric_h = (top_ref_height_px + bottom_ref_height_px) / 2 / REF_HEIGHT

    return pixels_per_metric_w, pixels_per_metric_h

# Main loop
red = [0, 0, 240]  # BGR format
green = [0, 150, 0]  # BGR format
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame
    processed_frame, red_masks, green_masks, ref_objects = process_frame(frame, red, green)
    
    # Show all processing steps for both colors
    cv2.imshow('Processed Frame', processed_frame)
    
    # Show red processing steps
    cv2.imshow('Red Original Mask', red_masks['original_mask'])
    cv2.imshow('Red Final Mask', red_masks['closed_mask'])
    
    # Show green processing steps
    cv2.imshow('Green Original Mask', green_masks['original_mask'])
    cv2.imshow('Green Final Mask', green_masks['closed_mask'])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()