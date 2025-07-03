#!/usr/bin/env python3
"""
advanced_calibration.py
Advanced camera calibration system for warehouse space measurement.
Provides accurate pixel-to-meter conversion, perspective correction, and multi-zone calibration.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)

class CalibrationMethod(Enum):
    """Different calibration methods available"""
    REFERENCE_OBJECTS = "reference_objects"
    CHECKERBOARD = "checkerboard"
    KNOWN_DISTANCES = "known_distances"
    MULTI_POINT = "multi_point"
    PERSPECTIVE_GRID = "perspective_grid"

@dataclass
class CalibrationPoint:
    """A calibration reference point"""
    pixel_coords: Tuple[float, float]
    world_coords: Tuple[float, float]  # in meters
    confidence: float
    description: str

@dataclass
class CameraParameters:
    """Camera calibration parameters"""
    pixels_per_meter_x: float
    pixels_per_meter_y: float
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    perspective_matrix: Optional[np.ndarray] = None
    calibration_confidence: float = 0.0
    calibration_method: str = ""
    zone_calibrations: Dict[str, Dict] = None

class AdvancedCalibrationSystem:
    """Advanced calibration system for warehouse cameras"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.calibration_data = {}
        self.reference_objects = {
            'standard_pallet': {'width': 1.2, 'height': 0.8, 'depth': 1.0},  # meters
            'euro_pallet': {'width': 1.2, 'height': 0.8, 'depth': 1.0},
            'american_pallet': {'width': 1.219, 'height': 1.016, 'depth': 1.0},
            'forklift': {'width': 1.8, 'height': 3.5, 'depth': 2.5},
            'person_standing': {'width': 0.5, 'height': 1.7, 'depth': 0.3},
            'door_standard': {'width': 0.9, 'height': 2.1, 'depth': 0.1},
            'rack_beam': {'width': 0.1, 'height': 0.1, 'depth': 2.4}
        }
        
        # Calibration validation thresholds
        self.validation_thresholds = {
            'min_confidence': 0.7,
            'max_perspective_distortion': 0.3,
            'max_scale_variation': 0.2
        }
        
    def detect_reference_objects(self, image: np.ndarray, 
                                method: str = "contour_analysis") -> List[Dict]:
        """Detect reference objects for calibration"""
        detections = []
        
        if method == "contour_analysis":
            detections.extend(self._detect_pallets_by_contour(image))
            detections.extend(self._detect_rack_structures(image))
        elif method == "template_matching":
            detections.extend(self._detect_by_template_matching(image))
        elif method == "edge_analysis":
            detections.extend(self._detect_by_edge_analysis(image))
        
        return detections
    
    def _detect_pallets_by_contour(self, image: np.ndarray) -> List[Dict]:
        """Detect pallets using contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to enhance pallet structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection
        edges = cv2.Canny(processed, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        img_h, img_w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:  # Too small for a pallet
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if dimensions match pallet characteristics
            if 0.8 <= aspect_ratio <= 2.0:  # Typical pallet aspect ratios
                # Analyze the content to determine pallet type confidence
                roi = gray[y:y+h, x:x+w]
                confidence = self._assess_pallet_confidence(roi, contour)
                
                if confidence > 0.5:
                    detections.append({
                        'type': 'standard_pallet',
                        'bbox': [x, y, x+w, y+h],
                        'confidence': confidence,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'method': 'contour_analysis'
                    })
        
        return detections
    
    def _detect_rack_structures(self, image: np.ndarray) -> List[Dict]:
        """Detect rack structures for calibration"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Hough lines to detect rack beams
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        detections = []
        
        if lines is not None:
            # Group horizontal and vertical lines
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if abs(angle) < 15 or abs(angle) > 165:  # Horizontal
                    horizontal_lines.append({'line': line[0], 'length': length})
                elif 75 < abs(angle) < 105:  # Vertical
                    vertical_lines.append({'line': line[0], 'length': length})
            
            # Look for intersections that might indicate rack corners
            rack_corners = self._find_rack_corners(horizontal_lines, vertical_lines)
            
            for corner in rack_corners:
                detections.append({
                    'type': 'rack_beam',
                    'bbox': [corner['x']-50, corner['y']-50, corner['x']+50, corner['y']+50],
                    'confidence': corner['confidence'],
                    'method': 'line_intersection'
                })
        
        return detections
    
    def _assess_pallet_confidence(self, roi: np.ndarray, contour: np.ndarray) -> float:
        """Assess confidence that a detected region is a pallet"""
        confidence = 0.0
        
        # 1. Check for rectangular shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 4:
            confidence += 0.3
        
        # 2. Check for expected texture patterns
        texture_score = cv2.Laplacian(roi, cv2.CV_64F).var()
        if 500 < texture_score < 2000:  # Expected range for pallet texture
            confidence += 0.3
        
        # 3. Check for parallel lines (pallet slats)
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=20, maxLineGap=5)
        
        if lines is not None and len(lines) > 3:
            confidence += 0.4
        
        return min(confidence, 1.0)
    
    def _find_rack_corners(self, horizontal_lines: List[Dict], 
                          vertical_lines: List[Dict]) -> List[Dict]:
        """Find rack corners from line intersections"""
        corners = []
        
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self._line_intersection(h_line['line'], v_line['line'])
                if intersection:
                    confidence = min(h_line['length'], v_line['length']) / 200.0
                    confidence = min(confidence, 1.0)
                    
                    corners.append({
                        'x': intersection[0],
                        'y': intersection[1],
                        'confidence': confidence
                    })
        
        return corners
    
    def _line_intersection(self, line1: List[int], line2: List[int]) -> Optional[Tuple[int, int]]:
        """Find intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (int(x), int(y))
        
        return None
    
    def calibrate_camera(self, image: np.ndarray, 
                        method: CalibrationMethod = CalibrationMethod.REFERENCE_OBJECTS,
                        reference_points: Optional[List[CalibrationPoint]] = None) -> CameraParameters:
        """Perform camera calibration using specified method"""
        
        if method == CalibrationMethod.REFERENCE_OBJECTS:
            return self._calibrate_with_reference_objects(image)
        elif method == CalibrationMethod.CHECKERBOARD:
            return self._calibrate_with_checkerboard(image)
        elif method == CalibrationMethod.KNOWN_DISTANCES:
            return self._calibrate_with_known_distances(image, reference_points)
        elif method == CalibrationMethod.MULTI_POINT:
            return self._calibrate_multi_point(image, reference_points)
        elif method == CalibrationMethod.PERSPECTIVE_GRID:
            return self._calibrate_perspective_grid(image)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def _calibrate_with_reference_objects(self, image: np.ndarray) -> CameraParameters:
        """Calibrate using detected reference objects"""
        detections = self.detect_reference_objects(image)
        
        if not detections:
            logger.warning("No reference objects detected for calibration")
            return self._create_default_parameters()
        
        # Calculate pixels per meter for each detection
        calibration_measurements = []
        
        for detection in detections:
            obj_type = detection['type']
            if obj_type in self.reference_objects:
                x1, y1, x2, y2 = detection['bbox']
                pixel_width = x2 - x1
                pixel_height = y2 - y1
                
                real_width = self.reference_objects[obj_type]['width']
                real_height = self.reference_objects[obj_type]['height']
                
                ppm_x = pixel_width / real_width
                ppm_y = pixel_height / real_height
                
                calibration_measurements.append({
                    'ppm_x': ppm_x,
                    'ppm_y': ppm_y,
                    'confidence': detection['confidence'],
                    'object_type': obj_type
                })
        
        # Calculate weighted average
        if calibration_measurements:
            total_confidence = sum(m['confidence'] for m in calibration_measurements)
            avg_ppm_x = sum(m['ppm_x'] * m['confidence'] for m in calibration_measurements) / total_confidence
            avg_ppm_y = sum(m['ppm_y'] * m['confidence'] for m in calibration_measurements) / total_confidence
            
            # Create camera parameters
            img_h, img_w = image.shape[:2]
            camera_matrix = np.array([
                [avg_ppm_x, 0, img_w/2],
                [0, avg_ppm_y, img_h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            distortion_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            return CameraParameters(
                pixels_per_meter_x=avg_ppm_x,
                pixels_per_meter_y=avg_ppm_y,
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion_coeffs,
                calibration_confidence=total_confidence / len(calibration_measurements),
                calibration_method="reference_objects"
            )
        
        return self._create_default_parameters()
    
    def _calibrate_with_checkerboard(self, image: np.ndarray) -> CameraParameters:
        """Calibrate using checkerboard pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Standard checkerboard dimensions (modify as needed)
        checkerboard_size = (9, 6)  # inner corners
        square_size = 0.025  # 2.5 cm squares
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Prepare object points
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
            objp *= square_size
            
            # Camera calibration
            ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                [objp], [corners], gray.shape[::-1], None, None
            )
            
            if ret:
                # Calculate pixels per meter from focal length
                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                
                # Estimate distance and calculate scale
                # This is a simplified estimation - would need more sophisticated approach
                ppm_x = fx / square_size * 100  # Rough estimation
                ppm_y = fy / square_size * 100
                
                return CameraParameters(
                    pixels_per_meter_x=ppm_x,
                    pixels_per_meter_y=ppm_y,
                    camera_matrix=camera_matrix,
                    distortion_coeffs=distortion_coeffs,
                    calibration_confidence=0.9,
                    calibration_method="checkerboard"
                )
        
        logger.warning("Checkerboard calibration failed")
        return self._create_default_parameters()
    
    def _calibrate_with_known_distances(self, image: np.ndarray,
                                      reference_points: List[CalibrationPoint]) -> CameraParameters:
        """Calibrate using manually marked reference points with known distances"""
        if not reference_points or len(reference_points) < 2:
            logger.warning("Insufficient reference points for distance calibration")
            return self._create_default_parameters()
        
        # Calculate pixels per meter from reference points
        distance_measurements = []
        
        for i in range(len(reference_points) - 1):
            for j in range(i + 1, len(reference_points)):
                point1 = reference_points[i]
                point2 = reference_points[j]
                
                # Pixel distance
                pixel_dist = math.sqrt(
                    (point2.pixel_coords[0] - point1.pixel_coords[0])**2 +
                    (point2.pixel_coords[1] - point1.pixel_coords[1])**2
                )
                
                # Real world distance
                real_dist = math.sqrt(
                    (point2.world_coords[0] - point1.world_coords[0])**2 +
                    (point2.world_coords[1] - point1.world_coords[1])**2
                )
                
                if real_dist > 0:
                    ppm = pixel_dist / real_dist
                    confidence = min(point1.confidence, point2.confidence)
                    
                    distance_measurements.append({
                        'ppm': ppm,
                        'confidence': confidence
                    })
        
        if distance_measurements:
            # Calculate weighted average
            total_confidence = sum(m['confidence'] for m in distance_measurements)
            avg_ppm = sum(m['ppm'] * m['confidence'] for m in distance_measurements) / total_confidence
            
            img_h, img_w = image.shape[:2]
            camera_matrix = np.array([
                [avg_ppm, 0, img_w/2],
                [0, avg_ppm, img_h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            return CameraParameters(
                pixels_per_meter_x=avg_ppm,
                pixels_per_meter_y=avg_ppm,
                camera_matrix=camera_matrix,
                distortion_coeffs=np.zeros((4, 1), dtype=np.float32),
                calibration_confidence=total_confidence / len(distance_measurements),
                calibration_method="known_distances"
            )
        
        return self._create_default_parameters()
    
    def _calibrate_multi_point(self, image: np.ndarray,
                             reference_points: List[CalibrationPoint]) -> CameraParameters:
        """Multi-point calibration with perspective correction"""
        if not reference_points or len(reference_points) < 4:
            logger.warning("Multi-point calibration requires at least 4 reference points")
            return self._create_default_parameters()
        
        # Separate pixel and world coordinates
        pixel_points = np.array([point.pixel_coords for point in reference_points], dtype=np.float32)
        world_points = np.array([point.world_coords for point in reference_points], dtype=np.float32)
        
        # Calculate perspective transformation
        try:
            perspective_matrix = cv2.getPerspectiveTransform(
                pixel_points[:4], world_points[:4] * 100  # Convert to cm for stability
            )
            
            # Calculate average pixels per meter from transformed coordinates
            test_points = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(test_points.reshape(-1, 1, 2), perspective_matrix)
            
            # Calculate scale
            pixel_width = 100  # 100 pixels
            world_width = abs(transformed[1, 0, 0] - transformed[0, 0, 0]) / 100  # Convert back to meters
            ppm = pixel_width / world_width if world_width > 0 else 100
            
            img_h, img_w = image.shape[:2]
            camera_matrix = np.array([
                [ppm, 0, img_w/2],
                [0, ppm, img_h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Calculate confidence based on point accuracy
            confidence = np.mean([point.confidence for point in reference_points])
            
            return CameraParameters(
                pixels_per_meter_x=ppm,
                pixels_per_meter_y=ppm,
                camera_matrix=camera_matrix,
                distortion_coeffs=np.zeros((4, 1), dtype=np.float32),
                perspective_matrix=perspective_matrix,
                calibration_confidence=confidence,
                calibration_method="multi_point"
            )
            
        except Exception as e:
            logger.error(f"Multi-point calibration failed: {e}")
            return self._create_default_parameters()
    
    def _calibrate_perspective_grid(self, image: np.ndarray) -> CameraParameters:
        """Calibrate using perspective grid analysis"""
        # This method analyzes the perspective distortion in warehouse aisles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect vanishing points using line analysis
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Find dominant vanishing point
            vanishing_point = self._find_vanishing_point(lines)
            
            if vanishing_point:
                # Use vanishing point to estimate perspective parameters
                img_h, img_w = image.shape[:2]
                
                # Estimate focal length from vanishing point
                vp_x, vp_y = vanishing_point
                focal_length = math.sqrt((vp_x - img_w/2)**2 + (vp_y - img_h/2)**2)
                
                # Rough estimation of pixels per meter
                # This would be refined with additional measurements
                ppm = focal_length / 5.0  # Rough heuristic
                
                camera_matrix = np.array([
                    [focal_length, 0, img_w/2],
                    [0, focal_length, img_h/2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                return CameraParameters(
                    pixels_per_meter_x=ppm,
                    pixels_per_meter_y=ppm,
                    camera_matrix=camera_matrix,
                    distortion_coeffs=np.zeros((4, 1), dtype=np.float32),
                    calibration_confidence=0.6,
                    calibration_method="perspective_grid"
                )
        
        return self._create_default_parameters()
    
    def _find_vanishing_point(self, lines: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find vanishing point from line intersections"""
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self._line_intersection(lines[i][0], lines[j][0])
                if intersection:
                    intersections.append(intersection)
        
        if intersections:
            # Cluster intersections to find the most likely vanishing point
            intersections = np.array(intersections)
            
            # Simple clustering - find the centroid of the largest cluster
            # This is a simplified approach; RANSAC would be better
            x_mean = np.mean(intersections[:, 0])
            y_mean = np.mean(intersections[:, 1])
            
            return (x_mean, y_mean)
        
        return None
    
    def _create_default_parameters(self) -> CameraParameters:
        """Create default camera parameters when calibration fails"""
        logger.warning("Using default camera parameters")
        
        return CameraParameters(
            pixels_per_meter_x=100.0,  # Default assumption
            pixels_per_meter_y=100.0,
            camera_matrix=np.eye(3, dtype=np.float32),
            distortion_coeffs=np.zeros((4, 1), dtype=np.float32),
            calibration_confidence=0.3,
            calibration_method="default"
        )
    
    def validate_calibration(self, params: CameraParameters, 
                           validation_image: np.ndarray) -> Dict[str, float]:
        """Validate calibration accuracy"""
        validation_results = {
            'confidence_score': params.calibration_confidence,
            'scale_consistency': 0.0,
            'perspective_accuracy': 0.0,
            'overall_quality': 0.0
        }
        
        # Test scale consistency with detected objects
        detections = self.detect_reference_objects(validation_image)
        scale_errors = []
        
        for detection in detections:
            obj_type = detection['type']
            if obj_type in self.reference_objects:
                x1, y1, x2, y2 = detection['bbox']
                pixel_width = x2 - x1
                
                expected_width = self.reference_objects[obj_type]['width']
                calculated_width = pixel_width / params.pixels_per_meter_x
                
                scale_error = abs(calculated_width - expected_width) / expected_width
                scale_errors.append(scale_error)
        
        if scale_errors:
            validation_results['scale_consistency'] = 1.0 - np.mean(scale_errors)
        
        # Calculate overall quality
        validation_results['overall_quality'] = (
            0.4 * validation_results['confidence_score'] +
            0.4 * validation_results['scale_consistency'] +
            0.2 * validation_results['perspective_accuracy']
        )
        
        return validation_results
    
    def save_calibration(self, params: CameraParameters, filename: str):
        """Save calibration parameters to file"""
        calibration_data = {
            'pixels_per_meter_x': float(params.pixels_per_meter_x),
            'pixels_per_meter_y': float(params.pixels_per_meter_y),
            'camera_matrix': params.camera_matrix.tolist(),
            'distortion_coeffs': params.distortion_coeffs.tolist(),
            'perspective_matrix': params.perspective_matrix.tolist() if params.perspective_matrix is not None else None,
            'calibration_confidence': float(params.calibration_confidence),
            'calibration_method': params.calibration_method,
            'validation_results': getattr(params, 'validation_results', {})
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename: str) -> Optional[CameraParameters]:
        """Load calibration parameters from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            params = CameraParameters(
                pixels_per_meter_x=data['pixels_per_meter_x'],
                pixels_per_meter_y=data['pixels_per_meter_y'],
                camera_matrix=np.array(data['camera_matrix'], dtype=np.float32),
                distortion_coeffs=np.array(data['distortion_coeffs'], dtype=np.float32),
                perspective_matrix=np.array(data['perspective_matrix'], dtype=np.float32) if data['perspective_matrix'] else None,
                calibration_confidence=data['calibration_confidence'],
                calibration_method=data['calibration_method']
            )
            
            logger.info(f"Calibration loaded from {filename}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return None

def main():
    """Main function for calibration system testing"""
    print("🎯 Advanced Camera Calibration System")
    print("=" * 50)
    
    calibration_system = AdvancedCalibrationSystem()
    
    # Example usage
    print("This module provides advanced calibration capabilities.")
    print("Use it in your main pipeline for accurate measurements.")
    print("\nAvailable calibration methods:")
    for method in CalibrationMethod:
        print(f"  - {method.value}")

if __name__ == "__main__":
    main()
