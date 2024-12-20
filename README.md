# Window Measurement Application

A computer vision application that measures window dimensions using credit cards as reference objects. The application uses color detection and computer vision techniques to automatically detect and measure window dimensions in centimeters from images.

## Overview

This application processes images containing two credit cards (marked in red) placed at different positions on a window. By using the credit cards' known dimensions as references, it calculates and displays the window's actual dimensions in centimeters.

## Requirements

* Python 3.x
* OpenCV (cv2)
* NumPy
* PIL (Python Imaging Library)

## Installation

1. Install the required packages:
   ```bash
   pip install opencv-python numpy pillow
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your directory structure:
   * Create a `calibration` folder for camera calibration images
   * Create folders for test images (e.g., `test_imgs`, `east_facing`, `west_facing`)

## How to Use

1. Place credit card shaped reference items on the window:
   * One in the top left
   * Another in the bottom left
   * Both cards should be clearly visible in the image

2. Organize your images:
   * Place calibration images in the `calibration` directory
   * Place test images in your designated test images directory

3. Run the application:
   ```python
   from window_measurer import WindowMeasurer

   measurer = WindowMeasurer(
       calibration_path='path/to/calibration',
       test_images_path='path/to/test/images'
   )
   measurer.run_measurement()
   ```
This is all in the `color_detector.py` file.

4. The application will:
   * Process each image in the test directory
   * Display color detection masks
   * Show measurements overlaid on the images
   * Press 'q' to move to the next image or quit

## Configuration

You can adjust the following parameters:

* `REF_WIDTH`: Credit card width (default: 8.56 cm)
* `REF_HEIGHT`: Credit card height (default: 5.398 cm)
* `min_area`: Minimum area for color detection (default: 3500 pixels)
* `kernel_size`: Size of kernel for morphological operations (default: 3)
* Color values in BGR format:
  * Red: `[95, 92, 201]`

## Release Notes

### Current Features

* Image-based window measurement processing
* Color detection for reference objects (credit cards)
* Reference object-based measurement calculation
* Measurement display in centimeters
* Noise reduction using morphological operations
* Camera calibration support
* Processing stage visualization:
  * Original masks
  * Eroded masks
  * Dilated masks
  * Opened masks
  * Closed masks
  * Final processed image with measurements

### Working Components

1. **Color Detection System**
   * Detection of red reference objects
   * Multiple stages of mask processing
   * Area-based filtering
   * Contour detection and processing

2. **Measurement System**
   * Pixel to centimeter conversion
   * Ratio-based calculations
   * Consistency scoring
   * Detailed measurement diagnostics

### Known Limitations

* Requires consistent lighting for reliable color detection
* Both credit cards must be visible in the image
* Performance depends on image quality and lighting conditions
* Credit cards must be clearly visible and unobstructed

## Future Improvements

* Implement real-time webcam processing functionality
* Add support for multiple color markers
* Develop automatic camera calibration optimization
* Include error handling for various lighting conditions
* Add support for different reference object types
* Implement batch processing with result logging
* Create a user interface for parameter adjustment
* Add export functionality for measurement results
* Implement perspective correction
* Add support for measuring multiple windows in one image

## Troubleshooting

If you experience issues with color detection:

1. Check image lighting conditions
2. Verify credit cards are clearly visible
3. Adjust the color values in the code if needed
4. Try increasing the `min_area` value if getting false positives
5. Adjust `kernel_size` for different levels of noise reduction
6. Check the consistency score in the measurement diagnostics
7. Verify the calibration images if using camera calibration
