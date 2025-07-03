#!/usr/bin/env python3
"""
src/main.py
Enhanced warehouse video analysis system with computer vision capabilities.
This script processes warehouse surveillance footage to detect and analyze storage space.
"""

import os
import cv2
import numpy as np
import requests
import json
from datetime import datetime
from ultralytics import YOLO
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from space_measurement import EnhancedSpaceMeasurement, SpaceMeasurement
from advanced_calibration import AdvancedCalibrationSystem, CalibrationMethod
from enhanced_model_trainer import EnhancedWarehouseModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RackCondition(Enum):
    """Enumeration for rack conditions"""
    GOOD = "good"
    PARTIALLY_BLOCKED = "partially-blocked"
    DAMAGED = "damaged"
    INACCESSIBLE = "inaccessible"

class RackType(Enum):
    """Enumeration for rack types"""
    MAIN_RACK = "main-rack"
    SMALL_SHELF = "small-shelf"
    PALLET_RACK = "pallet-rack"
    STORAGE_BIN = "storage-bin"

@dataclass
class DetectionResult:
    """Data class for detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

@dataclass
class RackAnalysis:
    """Data class for rack analysis results"""
    aisle: str
    rack_id: str
    rack_type: RackType
    total_capacity: int
    occupied_space: int
    free_space: int
    condition: RackCondition
    confidence: float
    bbox: Tuple[int, int, int, int]
    notes: str = ""
    area_sqm: float = 0.0
    occupancy_percentage: float = 0.0

class WarehouseAnalyzer:
    def __init__(self, api_base_url="http://localhost:5001", confidence_threshold=0.5):
        self.api_base_url = api_base_url
        self.model = None
        self.video_path = None
        self.output_dir = "output"
        self.confidence_threshold = confidence_threshold
        
        # Initialize enhanced systems
        self.calibration_system = AdvancedCalibrationSystem()
        self.model_trainer = EnhancedWarehouseModelTrainer()
        self.calibration_params = None

        # Enhanced configuration
        self.rack_classes = {
            'rack': 0,
            'pallet': 1,
            'shelf': 2,
            'empty_space': 3,
            'obstruction': 4
        }

        # Measurement parameters
        self.pixels_per_meter = 100  # Will be calibrated based on video
        self.standard_pallet_size = (1.2, 0.8)  # meters (length, width)
        self.rack_capacity_map = {
            RackType.MAIN_RACK: 3,
            RackType.SMALL_SHELF: 1,
            RackType.PALLET_RACK: 6,
            RackType.STORAGE_BIN: 4
        }

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize enhanced space measurement system
        self.space_measurement = EnhancedSpaceMeasurement(pixels_per_meter=self.pixels_per_meter)
        
        # Try to load existing calibration
        self.load_calibration_if_available()

        logger.info(f"Initialized WarehouseAnalyzer with confidence threshold: {confidence_threshold}")
        
    def load_calibration_if_available(self):
        """Load existing calibration parameters if available"""
        calibration_file = "calibration/results/calibration.json"
        if os.path.exists(calibration_file):
            self.calibration_params = self.calibration_system.load_calibration(calibration_file)
            if self.calibration_params:
                self.pixels_per_meter = self.calibration_params.pixels_per_meter_x
                self.space_measurement = EnhancedSpaceMeasurement(pixels_per_meter=self.pixels_per_meter)
                logger.info(f"Loaded calibration: {self.pixels_per_meter:.2f} pixels/meter")
    
    def calibrate_camera(self, frame: np.ndarray) -> bool:
        """Perform camera calibration on a frame"""
        try:
            logger.info("Performing camera calibration...")
            
            # Try multiple calibration methods
            methods = [
                CalibrationMethod.REFERENCE_OBJECTS,
                CalibrationMethod.PERSPECTIVE_GRID,
                CalibrationMethod.CHECKERBOARD
            ]
            
            best_params = None
            best_confidence = 0
            
            for method in methods:
                try:
                    params = self.calibration_system.calibrate_camera(frame, method)
                    if params.calibration_confidence > best_confidence:
                        best_params = params
                        best_confidence = params.calibration_confidence
                except Exception as e:
                    logger.warning(f"Calibration method {method.value} failed: {e}")
            
            if best_params and best_confidence > 0.5:
                self.calibration_params = best_params
                self.pixels_per_meter = best_params.pixels_per_meter_x
                self.space_measurement = EnhancedSpaceMeasurement(pixels_per_meter=self.pixels_per_meter)
                
                # Save calibration
                os.makedirs("calibration/results", exist_ok=True)
                self.calibration_system.save_calibration(best_params, "calibration/results/calibration.json")
                
                logger.info(f"Calibration successful: {self.pixels_per_meter:.2f} pixels/meter (confidence: {best_confidence:.2f})")
                return True
            else:
                logger.warning("Camera calibration failed - using default parameters")
                return False
                
        except Exception as e:
            logger.error(f"Camera calibration error: {e}")
            return False

    def load_model(self, model_path="models/warehouse_yolov8.pt"):
        """Load YOLO model for object detection"""
        try:
            # Try enhanced model first
            enhanced_model_path = "models/final/warehouse_best.pt"
            if os.path.exists(enhanced_model_path):
                print(f"Loading enhanced model from {enhanced_model_path}")
                self.model = YOLO(enhanced_model_path)
            elif os.path.exists(model_path):
                print(f"Loading custom model from {model_path}")
                self.model = YOLO(model_path)
            else:
                print("Custom model not found. Using pretrained YOLOv8 model.")
                self.model = YOLO('yolov8n.pt')  # Use pretrained YOLOv8 model
            
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def extract_frames(self, video_path, frame_interval=30):
        """Extract frames from video for analysis"""
        print(f"📹 Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return []

        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append((frame_count, frame))
                print(f"📸 Extracted frame {frame_count}/{total_frames}")

            frame_count += 1

        cap.release()
        print(f"✅ Extracted {len(frames)} frames for analysis")
        return frames

    def analyze_frame(self, frame, frame_number):
        """Analyze a single frame to detect storage areas using computer vision"""
        if self.model is None:
            logger.error("Model not loaded")
            return None

        try:
            # Run YOLO inference
            results = self.model(frame)

            # Process detections
            detections = self.process_detections(results[0])

            # Analyze racks and calculate space
            rack_analysis = self.analyze_racks_from_detections(frame, detections)

            analysis_result = {
                'frame_number': frame_number,
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'rack_analysis': rack_analysis
            }

            logger.info(f"Analyzed frame {frame_number}: found {len(rack_analysis)} racks")
            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_number}: {e}")
            return None

    def process_detections(self, results) -> List[DetectionResult]:
        """Process YOLO detection results into structured format"""
        detections = []

        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                # Extract detection data
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())

                # Filter by confidence threshold
                if confidence >= self.confidence_threshold:
                    # Map class ID to name
                    class_name = self.get_class_name(class_id)

                    detection = DetectionResult(
                        bbox=tuple(bbox),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    detections.append(detection)

        return detections

    def get_class_name(self, class_id: int) -> str:
        """Map class ID to human-readable name"""
        class_map = {
            0: 'rack',
            1: 'pallet',
            2: 'shelf',
            3: 'empty_space',
            4: 'obstruction'
        }
        return class_map.get(class_id, f'unknown_{class_id}')

    def analyze_racks_from_detections(self, frame, detections: List[DetectionResult]) -> List[Dict]:
        """Analyze rack occupancy from detection results"""
        rack_analyses = []

        # Group detections by spatial proximity to identify racks
        rack_detections = [d for d in detections if d.class_name in ['rack', 'shelf']]
        pallet_detections = [d for d in detections if d.class_name == 'pallet']
        obstruction_detections = [d for d in detections if d.class_name == 'obstruction']

        for rack_det in rack_detections:
            try:
                analysis = self.analyze_single_rack(frame, rack_det, pallet_detections, obstruction_detections)
                if analysis:
                    rack_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze rack: {e}")

        return rack_analyses

    def analyze_single_rack(self, frame, rack_detection: DetectionResult,
                           pallet_detections: List[DetectionResult],
                           obstruction_detections: List[DetectionResult]) -> Optional[Dict]:
        """Analyze a single rack for occupancy and condition using enhanced measurement"""
        try:
            # Convert detection results to format expected by space measurement
            pallet_dicts = [{'bbox': p.bbox, 'confidence': p.confidence} for p in pallet_detections]
            obstruction_dicts = [{'bbox': o.bbox, 'confidence': o.confidence} for o in obstruction_detections]

            # Use enhanced space measurement system with seasonal and damage assessment
            space_measurement = self.space_measurement.measure_rack_space(
                rack_detection.bbox, pallet_dicts, obstruction_dicts, frame,
                seasonal_mode="summer",  # Could be parameterized
                damage_assessment=True
            )

            # Determine rack type based on size and aspect ratio
            rack_type = self.classify_rack_type(rack_detection.bbox, frame.shape)

            # Determine condition based on accessibility and obstructions
            condition = self.assess_rack_condition_enhanced(space_measurement, rack_detection)

            # Generate rack ID and aisle
            rack_id, aisle = self.generate_rack_identifiers(rack_detection.bbox, frame.shape)

            return {
                'aisle': aisle,
                'rack_id': rack_id,
                'rack_type': rack_type.value,
                'total_capacity': space_measurement.capacity_units,
                'occupied_space': space_measurement.occupied_units,
                'free_space': space_measurement.free_units,
                'condition': condition.value,
                'confidence': rack_detection.confidence,
                'bbox': rack_detection.bbox,
                'area_sqm': space_measurement.total_area_sqm,
                'usable_area_sqm': space_measurement.usable_area_sqm,
                'occupancy_percentage': space_measurement.occupancy_percentage,
                'accessibility_score': space_measurement.accessibility_score,
                'measurement_confidence': space_measurement.measurement_confidence,
                'occupancy_level': space_measurement.occupancy_level.value,
                'obstructions_count': len(space_measurement.obstructions),
                'notes': space_measurement.notes
            }

        except Exception as e:
            logger.error(f"Error analyzing single rack: {e}")
            return self.create_fallback_rack_analysis(rack_detection, frame)

    def assess_rack_condition_enhanced(self, space_measurement: SpaceMeasurement,
                                     rack_detection: DetectionResult) -> RackCondition:
        """Enhanced rack condition assessment using space measurement data"""
        # Check accessibility score
        if space_measurement.accessibility_score < 0.3:
            return RackCondition.INACCESSIBLE
        elif space_measurement.accessibility_score < 0.6:
            return RackCondition.PARTIALLY_BLOCKED

        # Check for significant obstructions
        if len(space_measurement.obstructions) > 2:
            return RackCondition.PARTIALLY_BLOCKED

        # Check measurement confidence (low confidence might indicate damage)
        if space_measurement.measurement_confidence < 0.5:
            return RackCondition.DAMAGED

        # Check detection confidence
        if rack_detection.confidence < 0.6:
            return RackCondition.DAMAGED

        return RackCondition.GOOD

    def create_fallback_rack_analysis(self, rack_detection: DetectionResult, frame) -> Dict:
        """Create fallback analysis when enhanced measurement fails"""
        rack_type = self.classify_rack_type(rack_detection.bbox, frame.shape)
        rack_id, aisle = self.generate_rack_identifiers(rack_detection.bbox, frame.shape)
        area_sqm = self.calculate_area_sqm(rack_detection.bbox)

        return {
            'aisle': aisle,
            'rack_id': rack_id,
            'rack_type': rack_type.value,
            'total_capacity': self.rack_capacity_map.get(rack_type, 3),
            'occupied_space': 0,
            'free_space': self.rack_capacity_map.get(rack_type, 3),
            'condition': RackCondition.GOOD.value,
            'confidence': rack_detection.confidence,
            'bbox': rack_detection.bbox,
            'area_sqm': area_sqm,
            'usable_area_sqm': area_sqm,
            'occupancy_percentage': 0.0,
            'accessibility_score': 0.8,
            'measurement_confidence': 0.5,
            'occupancy_level': 'empty',
            'obstructions_count': 0,
            'notes': 'Fallback analysis - enhanced measurement failed'
        }

    def classify_rack_type(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> RackType:
        """Classify rack type based on size and aspect ratio"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        area = width * height
        frame_area = frame_shape[0] * frame_shape[1]
        relative_area = area / frame_area

        # Classification logic based on size and aspect ratio
        if relative_area > 0.3:  # Large area
            return RackType.PALLET_RACK
        elif aspect_ratio > 2.0:  # Wide and short
            return RackType.SMALL_SHELF
        elif relative_area > 0.1:  # Medium area
            return RackType.MAIN_RACK
        else:
            return RackType.STORAGE_BIN

    def simulate_rack_analysis(self, frame):
        """Fallback simulation for when real detection fails"""
        logger.warning("Using simulated rack analysis - consider training a proper model")

        height, width = frame.shape[:2]

        # Simulate detecting 2-3 racks in the frame with more realistic data
        simulated_racks = []
        for i in range(np.random.randint(1, 4)):
            rack_type = np.random.choice(list(RackType))
            condition = np.random.choice(list(RackCondition), p=[0.7, 0.2, 0.08, 0.02])

            total_capacity = self.rack_capacity_map.get(rack_type, 3)
            occupied_space = np.random.randint(0, total_capacity + 1)

            rack = {
                'aisle': f'A-{np.random.randint(1, 6)}',
                'rack_id': f'R-{np.random.randint(100, 999):03d}',
                'rack_type': rack_type.value,
                'total_capacity': total_capacity,
                'occupied_space': occupied_space,
                'free_space': max(0, total_capacity - occupied_space),
                'condition': condition.value,
                'confidence': np.random.uniform(0.6, 0.9),
                'bbox': [
                    np.random.randint(0, width//2),
                    np.random.randint(0, height//2),
                    np.random.randint(width//2, width),
                    np.random.randint(height//2, height)
                ],
                'area_sqm': np.random.uniform(2.0, 15.0),
                'occupancy_percentage': (occupied_space / total_capacity * 100) if total_capacity > 0 else 0,
                'notes': f"Simulated data - {condition.value} condition"
            }
            simulated_racks.append(rack)

        return simulated_racks

    def count_pallets_in_area(self, rack_bbox: Tuple[int, int, int, int],
                             pallet_detections: List[DetectionResult]) -> List[DetectionResult]:
        """Count pallets within a rack area"""
        rx1, ry1, rx2, ry2 = rack_bbox
        pallets_in_rack = []

        for pallet in pallet_detections:
            px1, py1, px2, py2 = pallet.bbox
            # Check if pallet center is within rack bounds
            pallet_center_x = (px1 + px2) / 2
            pallet_center_y = (py1 + py2) / 2

            if rx1 <= pallet_center_x <= rx2 and ry1 <= pallet_center_y <= ry2:
                pallets_in_rack.append(pallet)

        return pallets_in_rack

    def count_obstructions_in_area(self, rack_bbox: Tuple[int, int, int, int],
                                  obstruction_detections: List[DetectionResult]) -> List[DetectionResult]:
        """Count obstructions within a rack area"""
        rx1, ry1, rx2, ry2 = rack_bbox
        obstructions_in_rack = []

        for obstruction in obstruction_detections:
            ox1, oy1, ox2, oy2 = obstruction.bbox
            # Check for overlap
            if not (ox2 < rx1 or ox1 > rx2 or oy2 < ry1 or oy1 > ry2):
                obstructions_in_rack.append(obstruction)

        return obstructions_in_rack

    def calculate_rack_capacity(self, rack_type: RackType, bbox: Tuple[int, int, int, int]) -> int:
        """Calculate rack capacity based on type and size"""
        base_capacity = self.rack_capacity_map.get(rack_type, 3)

        # Adjust capacity based on actual size
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Scale capacity based on area (larger racks can hold more)
        if area > 50000:  # Very large rack
            return base_capacity + 2
        elif area > 20000:  # Large rack
            return base_capacity + 1
        elif area < 5000:  # Small rack
            return max(1, base_capacity - 1)

        return base_capacity

    def assess_rack_condition(self, rack_detection: DetectionResult,
                            obstructions: List[DetectionResult], frame) -> RackCondition:
        """Assess rack condition based on various factors"""
        # Check for obstructions
        if len(obstructions) > 0:
            obstruction_area = sum((o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) for o in obstructions)
            rack_area = (rack_detection.bbox[2] - rack_detection.bbox[0]) * (rack_detection.bbox[3] - rack_detection.bbox[1])

            if obstruction_area > rack_area * 0.5:  # More than 50% obstructed
                return RackCondition.INACCESSIBLE
            elif obstruction_area > rack_area * 0.2:  # More than 20% obstructed
                return RackCondition.PARTIALLY_BLOCKED

        # Check detection confidence
        if rack_detection.confidence < 0.6:
            return RackCondition.DAMAGED  # Low confidence might indicate damage

        # Additional checks could include:
        # - Image quality analysis
        # - Structural integrity assessment
        # - Lighting conditions

        return RackCondition.GOOD

    def calculate_area_sqm(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate area in square meters using pixel-to-meter conversion"""
        x1, y1, x2, y2 = bbox
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        # Convert to meters
        width_meters = width_pixels / self.pixels_per_meter
        height_meters = height_pixels / self.pixels_per_meter

        return width_meters * height_meters

    def generate_rack_identifiers(self, bbox: Tuple[int, int, int, int],
                                 frame_shape: Tuple[int, int]) -> Tuple[str, str]:
        """Generate rack ID and aisle based on position"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        frame_height, frame_width = frame_shape[:2]

        # Determine aisle based on horizontal position
        aisle_num = int((center_x / frame_width) * 5) + 1  # 5 aisles
        aisle = f"A-{aisle_num}"

        # Generate rack ID based on position
        rack_num = int((center_y / frame_height) * 20) + 100  # Start from R-100
        rack_id = f"R-{rack_num:03d}"

        return rack_id, aisle

    def generate_analysis_notes(self, condition: RackCondition,
                              obstructions: List[DetectionResult], confidence: float) -> str:
        """Generate descriptive notes for the analysis"""
        notes = []

        if condition == RackCondition.PARTIALLY_BLOCKED:
            notes.append(f"Partially blocked by {len(obstructions)} obstruction(s)")
        elif condition == RackCondition.DAMAGED:
            notes.append("Possible structural damage detected")
        elif condition == RackCondition.INACCESSIBLE:
            notes.append("Area inaccessible due to obstructions")

        if confidence < 0.7:
            notes.append(f"Low detection confidence ({confidence:.2f})")

        notes.append(f"Auto-detected (confidence: {confidence:.2f})")

        return "; ".join(notes) if notes else "Good condition"

    def send_to_api(self, rack_data):
        """Send analyzed rack data to the backend API"""
        try:
            for rack in rack_data:
                entry = {
                    "aisle": rack['aisle'],
                    "rackId": rack['rack_id'],
                    "rackType": rack['rack_type'],
                    "totalCapacity": rack['total_capacity'],
                    "occupiedSpace": rack['occupied_space'],
                    "freeSpace": rack['free_space'],
                    "condition": rack['condition'],
                    "notes": f"Auto-detected (confidence: {rack['confidence']:.2f})",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                response = requests.post(f"{self.api_base_url}/api/add", json=entry)
                if response.status_code == 200:
                    print(f"✅ Sent data for {rack['rack_id']} to API")
                else:
                    print(f"❌ Failed to send data for {rack['rack_id']}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending data to API: {e}")

    def save_annotated_frame(self, frame, analysis_result, frame_number):
        """Save frame with detection annotations"""
        annotated_frame = frame.copy()

        for rack in analysis_result['rack_analysis']:
            bbox = rack['bbox']
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            color = (0, 255, 0) if rack['condition'] == 'good' else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Add text labels
            label = f"{rack['rack_id']}: {rack['free_space']}/{rack['total_capacity']} free"
            cv2.putText(annotated_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save annotated frame
        output_path = os.path.join(self.output_dir, f"annotated_frame_{frame_number:06d}.jpg")
        cv2.imwrite(output_path, annotated_frame)
        print(f"💾 Saved annotated frame: {output_path}")

    def process_video(self, video_path, send_to_api=True, save_frames=True):
        """Main video processing pipeline"""
        print(f"🚀 Starting warehouse analysis for {video_path}")

        if not self.load_model():
            return False

        # Extract frames for analysis
        frames = self.extract_frames(video_path, frame_interval=60)  # Every 2 seconds at 30fps
        
        # Perform camera calibration on first frame if not already calibrated
        if not self.calibration_params and frames:
            self.calibrate_camera(frames[0][1])

        all_rack_data = []

        for frame_number, frame in frames:
            print(f"🔍 Analyzing frame {frame_number}")

            # Analyze the frame
            analysis_result = self.analyze_frame(frame, frame_number)

            if analysis_result and analysis_result['rack_analysis']:
                all_rack_data.extend(analysis_result['rack_analysis'])

                # Save annotated frame
                if save_frames:
                    self.save_annotated_frame(frame, analysis_result, frame_number)

        # Remove duplicates and send unique racks to API
        if send_to_api and all_rack_data:
            unique_racks = self.remove_duplicate_racks(all_rack_data)
            print(f"📊 Found {len(unique_racks)} unique racks")
            self.send_to_api(unique_racks)

        print("✅ Video analysis complete!")
        return True

    def remove_duplicate_racks(self, rack_data):
        """Remove duplicate rack detections based on rack_id"""
        seen_racks = {}
        for rack in rack_data:
            rack_id = rack['rack_id']
            if rack_id not in seen_racks or rack['confidence'] > seen_racks[rack_id]['confidence']:
                seen_racks[rack_id] = rack
        return list(seen_racks.values())


def main():
    """Main function to run the warehouse space analysis."""
    print("🏭 Warehouse Free Space Analysis System")
    print("=" * 50)

    # Initialize analyzer
    analyzer = WarehouseAnalyzer()

    # Get list of video files
    video_dir = "video"
    if not os.path.exists(video_dir):
        print(f"❌ Video directory '{video_dir}' not found")
        return

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print(f"❌ No video files found in '{video_dir}'")
        return

    print(f"📹 Found {len(video_files)} video files:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\n🎬 Processing: {video_file}")

        success = analyzer.process_video(video_path)
        if success:
            print(f"✅ Successfully processed {video_file}")
        else:
            print(f"❌ Failed to process {video_file}")

    print("\n🎉 All videos processed!")
    print(f"📁 Check the '{analyzer.output_dir}' directory for annotated frames")
    print(f"🌐 Visit http://localhost:5001 to view the dashboard")


if __name__ == "__main__":
    main()