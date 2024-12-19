"""
Author: Fabrice Bokovi
Description: Perform window measurement using color detection of the two reference objects placed on the window.
Version: 1.0
"""

import cv2
import numpy as np
from utils import get_limits
from PIL import Image
from camera_calibration import CameraCalibrator
from dataclasses import dataclass
import os
from typing import Optional, Tuple, List

class WindowMeasurer:
    def __init__(self, calibration_path: str, test_images_path: str):
        # Constants
        self.REF_WIDTH = 8.56  # credit card width in cm
        self.REF_HEIGHT = 5.398  # credit card height in cm
        self.RED = [95, 92, 201]  # BGR format
        
        # Paths
        self.cwd = os.getcwd()
        self.test_imgs_path = test_images_path
        self.calibration_path = calibration_path
        
        # Initialize calibrator
        self.calibrator = CameraCalibrator(
            showPics=True,
            cal_images_path=self.calibration_path
        )
        
        # self.calibrator.calibrate()
        
        # Take a picture
        # cap = cv2.VideoCapture(0)
        
        # # Set resolution
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # if cap.isOpened():
        #     _, frame = cap.read()
        #     cap.release()
        #     if _ and frame is not None:
        #         cv2.imwrite(os.path.join(self.test_imgs_path, 'test_img.jpg'), frame)
                
        # get test images
        self.test_imgs = [
            os.path.join(self.test_imgs_path, f) 
            for f in os.listdir(self.test_imgs_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        
    def process_frame(self, img_path: str, color:List[int], 
                      min_area: int = 3500, kernel_size: int = 3) -> Tuple[Optional[np.ndarray], dict, list]:
        """Process a single frame to detect and measure window"""
        
        try:
            # Read image
            if isinstance(img_path, str):
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Failed to read image from {img_path}")
            else:
                image = img_path
            
            if image is None:
                raise ValueError("Invalid image input")
            
            # Color detection and masking
            def detect_color(color):
                # Convert to HSV
                hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lowerLimit, upperLimit = get_limits(color=color)

                # Create mask
                mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
                
                # Morphological operations
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded_mask = cv2.erode(mask, kernel, iterations=1)
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
                 
            # Detect colors and find contours
            masks = detect_color(color)
            contours, _ = cv2.findContours(
                masks['closed_mask'], 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            ref_objects = []
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    ref_objects.append({
                        'position': '',
                        'coords': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
                    cv2.putText(image, f'Red: {int(area)}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if len(ref_objects) >= 2:
                # Sort largest 2 contours by vertical position
                ref_objects.sort(key=lambda x: x['area'], reverse=True)
                ref_objects = ref_objects[:2]
                ref_objects.sort(key=lambda x: x['center'][1])
                    
                top_ref = ref_objects[0]
                bottom_ref = ref_objects[1]
                
                # get pixel per metric           
                avg_width_ratio, avg_height_ratio, consistency_score = self.get_pixel_per_metric(ref_objects=ref_objects)
                
                # Draw window pane boundaries
                window_coords = {
                    'top_left': (top_ref['coords'][0], top_ref['coords'][1]),
                    'bottom_right': (bottom_ref['coords'][0] + bottom_ref['coords'][2],
                                    bottom_ref['coords'][1] + bottom_ref['coords'][3])
                }
                
                # Calculate window dimensions in pixels
                window_width_px = window_coords['bottom_right'][0] - window_coords['top_left'][0]
                window_height_px = window_coords['bottom_right'][1] - window_coords['top_left'][1]
                
                print(f"Window height px: {window_height_px}px")
                print(f"Window width px: {window_width_px}px")
                
                # Calculate measurements using both ratios
                # width based calculations
                window_width_cm = window_width_px * avg_width_ratio
                window_height_cm = window_height_px * avg_width_ratio
                
                print(f"Window height cm: {window_height_cm:.02f}cm")
                print(f"Window width cm: {window_width_cm:.02f}cm")
                
                # heihgt based calculations
                height_based_width = window_width_px * avg_height_ratio
                height_based_height = window_height_px * avg_height_ratio
                
                # Draw window outline
                cv2.rectangle(image,
                            window_coords['top_left'],
                            window_coords['bottom_right'],
                            (255, 0, 0), 1)
                                
                
                
                # Display dimensions (metric)
                cv2.putText(
                    image,
                    f'Width: {window_width_cm:.2f}cm',
                    (window_coords['top_left'][0], window_coords['top_left'][1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (105, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f'Height: {window_height_cm:.2f}cm',
                    (window_coords['top_left'][0], window_coords['top_left'][1] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (105, 255, 0),
                    2
                )
                
            return image, masks, ref_objects

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, None, None
    
    def run_measurement(self, force_calibration=False):
        """Run the window measurement process"""
        try:
            # Perform/load calibration
            if not self.calibrator.load_calibration() or force_calibration:
                print("Running new calibration...")
                result = self.calibrator.calibrate()
                if not result:
                    raise ValueError("Calibration failed.")
                
            print("Processing iamges...")
            for idx, img_path in enumerate(self.test_imgs):
                print(f"\nProcessing image {idx + 1}/{len(self.test_imgs)}")
                
                # undistort images
                # undistorted = self.calibrator.remove_distortion(img_path)
                # if undistorted is None:
                #     print(f"Failed to undistort image: {img_path}")
                #     continue
                
                # Process frame
                processed_frame, masks, ref_objects = self.process_frame(img_path, self.RED)
                
                # show the masks
                for mask_key, mask_val in masks.items():
                    cv2.imshow(mask_key, mask_val)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    
                if processed_frame is not None:
                    cv2.imshow(f'Measurement {idx + 1}', processed_frame)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    continue
            
        except Exception as e:
            print(f"Error occured while running measurement: {e}")
        finally:
            cv2.destroyAllWindows()
            
    def get_pixel_per_metric(self, ref_objects):
        """Calculate pixels per metric conversion factors"""
        top_ref = ref_objects[0]
        bottom_ref = ref_objects[1]
        
        # Get dimensions in pixels
        top_ref_width_px = top_ref['coords'][2]
        top_ref_height_px = top_ref['coords'][3]
        bottom_ref_width_px = bottom_ref['coords'][2]
        bottom_ref_height_px = bottom_ref['coords'][3]
        
        # Calculate individual ratios
        top_width_ratio = self.REF_WIDTH / top_ref_width_px
        bottom_width_ratio = self.REF_WIDTH / bottom_ref_width_px
        top_height_ratio = self.REF_HEIGHT / top_ref_height_px
        bottom_height_ratio = self.REF_HEIGHT / bottom_ref_height_px
        
        # Calculate averages
        avg_width_ratio = (top_width_ratio + bottom_width_ratio) / 2
        avg_height_ratio = (top_height_ratio + bottom_height_ratio) / 2
        
        # Calculate consistency (how similar the ratios are)
        width_variance = abs(top_width_ratio - bottom_width_ratio) / avg_width_ratio
        height_variance = abs(top_height_ratio - bottom_height_ratio) / avg_height_ratio
        consistency_score = 1 - (width_variance + height_variance) / 2
        
        # Print detailed diagnostics
        print(f"\nMeasurement Diagnostics:")
        print(f"Reference object dimensions (pixels):")
        print(f"Top: {top_ref_width_px}x{top_ref_height_px}")
        print(f"Bottom: {bottom_ref_width_px}x{bottom_ref_height_px}")
        print(f"Ratios - Width: {avg_width_ratio:.4f}, Height: {avg_height_ratio:.4f}")
        print(f"Consistency score: {consistency_score:.4f}")

        return avg_width_ratio, avg_height_ratio, consistency_score

def main():
    # init paths
    cwd = os.getcwd()
    calibration_path = os.path.join(cwd, 'OpenCV Tutorial', 'calibration')
    # test_images_path = os.path.join(cwd, 'OpenCV Tutorial', 'test_imgs')
    # test_images_path = os.path.join(cwd, 'OpenCV Tutorial', 'west_facing')
    test_images_path = os.path.join(cwd, 'OpenCV Tutorial', 'east_facing')
    # test_images_path = os.path.join(cwd, 'OpenCV Tutorial', 'east_facing', 'WIN_20241205_13_28_36_Pro.jpg')
    # test_images_path = os.path.join(cwd, 'OpenCV Tutorial', 'uploads')

    # create measurer object
    measurer = WindowMeasurer(
        calibration_path=calibration_path, 
        test_images_path=test_images_path
    )
    
    # Run measurement
    measurer.run_measurement(force_calibration=False)
    
if __name__ == '__main__':
    main()