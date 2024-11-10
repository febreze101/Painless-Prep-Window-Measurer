# Window Measurement Application

A real-time computer vision application that measures window dimensions using a credit card sized reference object. The application uses color detection and computer vision techniques to automatically detect and measure window dimensions in centimeters.

## Overview

This application uses your computer's webcam to detect two colored markers (red) placed at opposite corners of a window. By using a credit card's known dimensions as a reference, it calculates and displays the window's actual dimensions in centimeters in real-time.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)

## Installation

1. Install the required packages:
```bash
pip install opencv-python numpy pillow
```

2. Ensure you have a working webcam connected to your computer.

## How to Use

1. Place colored markers at two opposite corners of the window:
   - One marker at the upper-left corner
   - Other marker at the bottom-right corner
   - The markers should be approximately the size of a credit card

2. Run the application:
```bash
python color_detection.py
```

3. Position yourself so that both markers are clearly visible in the webcam feed.

4. The application will:
   - Detect both colored markers
   - Draw rectangles around the detected markers
   - Display the window dimensions in centimeters
   - Show various processing stages in separate windows

5. Press 'q' to quit the application

## Configuration

You can adjust the following parameters in the code:

- `REF_WIDTH`: Reference object width (default: 8.56 cm for credit card)
- `REF_HEIGHT`: Reference object height (default: 5.398 cm for credit card)
- `min_area`: Minimum area for color detection (default: 3500 pixels)
- `kernel_size`: Size of kernel for morphological operations (default: 3)
- Color values for markers (in BGR format):
  - Red: `[0, 0, 240]`
  - Green: `[0, 150, 0]`

## Release Notes

### Current Features
- Real-time window measurement using webcam feed
- Color detection for red and green markers
- Reference object-based measurement calculation
- Live dimension display in centimeters
- Noise reduction using morphological operations
- Visualization of processing stages:
  - Original color masks
  - Final processed masks
  - Detected markers with area values
  - Window outline with measurements

### Working Components
1. **Color Detection System**
   - Successful detection of red and green markers
   - Robust noise filtering
   - Area-based filtering to reduce false positives

2. **Measurement System**
   - Conversion from pixels to centimeters
   - Dynamic calculation based on reference object dimensions
   - Real-time measurement updates

### Known Limitations
- Requires consistent lighting for reliable color detection
- Both markers must be visible simultaneously
- Markers must be approximately credit card sized for accurate measurements
- Performance may vary based on webcam quality and lighting conditions

## Troubleshooting

If you experience issues with color detection:
1. Ensure adequate lighting in the room
2. Check that markers are clearly visible and unobstructed
3. Adjust the color values in the code if needed
4. Try increasing the `min_area` value if getting false positives
5. Adjust `kernel_size` for different levels of noise reduction
