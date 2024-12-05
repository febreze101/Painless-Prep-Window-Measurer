import numpy as np
import cv2 as cv
import glob
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

@dataclass
class CalibrationResult:
    """Store calibration results"""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float
    success: bool

class CameraCalibrator:
    def __init__(self, showPics: bool = False, cal_images_path: Optional[str] = None) -> None:
        """
        Initialize Camera Calibrator
        
        Args:
            showPics: Whether to show calibration images
            cal_images_path: Path to calibration images directory
        """
        self.cwd = os.getcwd()
        self.showPics = showPics
        self.cal_images_path = cal_images_path
        self.chessboard_size = (8, 6)  # Move to class attribute
        self.calibration_result: Optional[CalibrationResult] = None

    def _find_chessboard_corners(self, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Find chessboard corners in a single image
        
        Args:
            image_path: Path to calibration image
            
        Returns:
            Tuple of success flag and corners if found
        """
        try:
            img_bgr = cv.imread(image_path)
            if img_bgr is None:
                print(f"Failed to read image: {image_path}")
                return False, None

            img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
            nRows, nCols = self.chessboard_size
            
            # Find corners
            corners_found, corners_org = cv.findChessboardCorners(
                img_gray, 
                (nRows, nCols), 
                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if corners_found:
                # Refine corners
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv.cornerSubPix(
                    img_gray, 
                    corners_org, 
                    (11, 11), 
                    (-1, -1), 
                    criteria
                )
                
                if self.showPics:
                    vis_img = img_bgr.copy()
                    cv.drawChessboardCorners(vis_img, (nRows, nCols), corners_refined, corners_found)
                    cv.imshow('Chessboard', vis_img)
                    cv.waitKey(500)
                
                return True, corners_refined
                
            return False, None
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False, None

    def calibrate(self) -> CalibrationResult:
        """
        Perform camera calibration
        
        Returns:
            CalibrationResult object containing calibration data
        """
        try:
            # Get calibration images
            if self.cal_images_path:
                img_paths = glob.glob(os.path.join(self.cal_images_path, '*.jpg'))
                if not img_paths:
                    raise ValueError(f"No jpg images found in {self.cal_images_path}")
            else:
                img_paths = glob.glob(os.path.join(self.cwd, 'demoImages', '*.jpg'))
            
            print(f"Found {len(img_paths)} calibration images")
            
            # Initialize calibration parameters
            nRows, nCols = self.chessboard_size
            world_points = np.zeros((nRows*nCols, 3), np.float32)
            world_points[:,:2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
            
            world_points_list = []
            img_points_list = []
            
            # Process each image
            successful_images = 0
            for img_path in img_paths:
                success, corners = self._find_chessboard_corners(img_path)
                if success:
                    successful_images += 1
                    world_points_list.append(world_points)
                    img_points_list.append(corners)
            
            print(f"Successfully processed {successful_images}/{len(img_paths)} images")
            
            if successful_images == 0:
                raise ValueError("No valid calibration patterns found")
            
            # Perform calibration
            ret, cam_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
                world_points_list,
                img_points_list,
                cv.imread(img_paths[0]).shape[:2][::-1],
                None,
                None
            )
            
            # Save results
            self.save_calibration(cam_matrix, dist_coeffs)
            
            # Store and return results
            self.calibration_result = CalibrationResult(
                camera_matrix=cam_matrix,
                dist_coeffs=dist_coeffs,
                reprojection_error=ret,
                success=True
            )
            
            return self.calibration_result
            
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            return CalibrationResult(None, None, float('inf'), False)

    def remove_distortion(self, image: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Remove distortion from an image using current calibration
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Undistorted image or None if failed
        """
        try:
            if self.calibration_result is None:
                raise ValueError("No calibration data available")
                
            # Handle input
            if isinstance(image, str):
                img = cv.imread(image)
                if img is None:
                    raise ValueError(f"Failed to read image: {image}")
            else:
                img = image
                
            # Get optimal camera matrix
            height, width = img.shape[:2]
            new_cam_matrix, roi = cv.getOptimalNewCameraMatrix(
                self.calibration_result.camera_matrix,
                self.calibration_result.dist_coeffs,
                (width, height),
                1,
                (width, height)
            )
            
            # Undistort
            return cv.undistort(
                img,
                self.calibration_result.camera_matrix,
                self.calibration_result.dist_coeffs,
                None,
                new_cam_matrix
            )
            
        except Exception as e:
            print(f"Error removing distortion: {str(e)}")
            return None

    def save_calibration(self, cam_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """Save calibration data to file"""
        try:
            save_path = os.path.join(self.cwd, 'OpenCV Tutorial', 'calibration_data.npz')
            np.savez(
                save_path,
                camera_matrix=cam_matrix,
                dist_coeffs=dist_coeffs
            )
            print(f"Calibration saved to {save_path}")
        except Exception as e:
            print(f"Error saving calibration: {str(e)}")

    def load_calibration(self) -> bool:
        try:
            load_path = os.path.join(self.cwd, 'OpenCV Tutorial', 'calibration_data.npz')
            if os.path.exists(load_path):
                data = np.load(load_path)
                dist_coeffs = data.get('dist_coeffs')
                if dist_coeffs is not None:
                    dist_coeffs = dist_coeffs.reshape(-1)  # Ensure correct shape
                
                self.calibration_result = CalibrationResult(
                    camera_matrix=data.get('camera_matrix'),
                    dist_coeffs=dist_coeffs,
                    reprojection_error=0.0,
                    success=True
                )
                return True
            return False
        except Exception as e:
            print(f"Error loading calibration: {str(e)}")
            return False
        
def main():
    # Initialize calibrator
    calibrator = CameraCalibrator(
        showPics=True,
        cal_images_path='path/to/calibration/images'
    )

    # Try to load existing calibration
    if not calibrator.load_calibration():
        # Perform new calibration
        result = calibrator.calibrate()
        if not result.success:
            print("Calibration failed")
            exit()

    # Use calibrator
    undistorted = calibrator.remove_distortion('path/to/image.jpg')
    if undistorted is not None:
        cv.imshow('Undistorted', undistorted)
        cv.waitKey(0)
    
if __name__ == "__main__":
    main()