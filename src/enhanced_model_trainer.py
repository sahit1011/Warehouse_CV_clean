#!/usr/bin/env python3
"""
enhanced_model_trainer.py
Advanced warehouse-specific YOLO model training system with comprehensive data preparation,
annotation tools, and robust training pipeline for accurate space detection.
"""

import os
import cv2
import numpy as np
import yaml
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnnotationType(Enum):
    """Types of annotations for warehouse objects"""
    RACK = "rack"
    PALLET = "pallet"
    EMPTY_SLOT = "empty_slot"
    OCCUPIED_SLOT = "occupied_slot"
    OBSTRUCTION = "obstruction"
    FORKLIFT = "forklift"
    PERSON = "person"

@dataclass
class Annotation:
    """Annotation data structure"""
    class_id: int
    bbox: Tuple[float, float, float, float]  # normalized coordinates
    confidence: float = 1.0
    metadata: Dict = None

class EnhancedWarehouseModelTrainer:
    """Advanced warehouse model training system"""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.setup_directories()
        
        # Class mapping for warehouse objects
        self.class_mapping = {
            AnnotationType.RACK: 0,
            AnnotationType.PALLET: 1,
            AnnotationType.EMPTY_SLOT: 2,
            AnnotationType.OCCUPIED_SLOT: 3,
            AnnotationType.OBSTRUCTION: 4,
            AnnotationType.FORKLIFT: 5,
            AnnotationType.PERSON: 6
        }
        
        # Training configuration
        self.training_config = {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'patience': 20,
            'data_augmentation': True,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1
        }
        
        # Camera calibration parameters
        self.calibration_params = {
            'pixels_per_meter': None,
            'perspective_matrix': None,
            'distortion_coeffs': None,
            'camera_matrix': None
        }
        
    def setup_directories(self):
        """Setup required directory structure"""
        dirs = [
            'data/images/train',
            'data/images/val',
            'data/images/test',
            'data/labels/train',
            'data/labels/val',
            'data/labels/test',
            'data/raw_videos',
            'data/annotations',
            'models/checkpoints',
            'models/final',
            'calibration/reference_images',
            'calibration/results'
        ]
        
        for dir_path in dirs:
            (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info("Directory structure created successfully")
    
    def extract_diverse_frames(self, video_path: str, 
                              frame_count: int = 500,
                              quality_threshold: float = 0.3) -> List[str]:
        """Extract diverse, high-quality frames for training"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame intervals for diverse sampling
        intervals = self._calculate_sampling_intervals(total_frames, frame_count)
        
        extracted_frames = []
        frame_quality_scores = []
        
        for frame_idx in intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Assess frame quality
            quality_score = self._assess_frame_quality(frame)
            
            if quality_score >= quality_threshold:
                # Save frame
                frame_filename = f"frame_{Path(video_path).stem}_{frame_idx:06d}.jpg"
                frame_path = self.project_root / 'data' / 'images' / 'train' / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(str(frame_path))
                frame_quality_scores.append(quality_score)
                
                logger.info(f"Extracted frame {frame_idx} with quality {quality_score:.3f}")
        
        cap.release()
        
        # Sort by quality and keep best frames
        if len(extracted_frames) > frame_count:
            sorted_frames = sorted(zip(extracted_frames, frame_quality_scores), 
                                 key=lambda x: x[1], reverse=True)
            extracted_frames = [f[0] for f in sorted_frames[:frame_count]]
        
        logger.info(f"Extracted {len(extracted_frames)} high-quality frames")
        return extracted_frames
    
    def _calculate_sampling_intervals(self, total_frames: int, target_count: int) -> List[int]:
        """Calculate optimal frame sampling intervals"""
        if total_frames <= target_count:
            return list(range(total_frames))
        
        # Use a combination of uniform and random sampling
        uniform_interval = total_frames // target_count
        uniform_frames = list(range(0, total_frames, uniform_interval))[:target_count//2]
        
        # Add random frames for diversity
        remaining_count = target_count - len(uniform_frames)
        excluded_ranges = [(f-uniform_interval//4, f+uniform_interval//4) for f in uniform_frames]
        
        random_frames = []
        for _ in range(remaining_count):
            while True:
                frame_idx = random.randint(0, total_frames-1)
                if not any(start <= frame_idx <= end for start, end in excluded_ranges):
                    random_frames.append(frame_idx)
                    break
        
        return sorted(uniform_frames + random_frames)
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality based on multiple criteria"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # 2. Brightness consistency
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5
        
        # 3. Contrast
        contrast_score = gray.std() / 128.0
        contrast_score = min(contrast_score, 1.0)
        
        # 4. Edge density (indicates detail richness)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density * 10, 1.0)
        
        # Combined quality score
        quality_score = (0.4 * sharpness_score + 
                        0.2 * brightness_score + 
                        0.2 * contrast_score + 
                        0.2 * edge_score)
        
        return quality_score
    
    def setup_camera_calibration(self, reference_images: List[str], 
                                known_dimensions: Dict[str, float]) -> bool:
        """Setup camera calibration for accurate space measurement"""
        try:
            # Use reference objects to calculate pixels per meter
            calibration_data = []
            
            for img_path in reference_images:
                if not os.path.exists(img_path):
                    continue
                    
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Detect reference objects (e.g., standard pallets)
                reference_detections = self._detect_reference_objects(image)
                
                for detection in reference_detections:
                    obj_type = detection['type']
                    if obj_type in known_dimensions:
                        pixel_width = detection['bbox'][2] - detection['bbox'][0]
                        pixel_height = detection['bbox'][3] - detection['bbox'][1]
                        
                        real_width = known_dimensions[obj_type]['width']
                        real_height = known_dimensions[obj_type]['height']
                        
                        pixels_per_meter_w = pixel_width / real_width
                        pixels_per_meter_h = pixel_height / real_height
                        
                        calibration_data.append({
                            'pixels_per_meter': (pixels_per_meter_w + pixels_per_meter_h) / 2,
                            'confidence': detection['confidence']
                        })
            
            if calibration_data:
                # Calculate weighted average
                total_confidence = sum(d['confidence'] for d in calibration_data)
                self.calibration_params['pixels_per_meter'] = sum(
                    d['pixels_per_meter'] * d['confidence'] for d in calibration_data
                ) / total_confidence
                
                # Save calibration
                self._save_calibration_data()
                logger.info(f"Calibration successful: {self.calibration_params['pixels_per_meter']:.2f} pixels/meter")
                return True
            
            logger.warning("Calibration failed - no reference objects detected")
            return False
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False
    
    def _detect_reference_objects(self, image: np.ndarray) -> List[Dict]:
        """Detect reference objects for calibration (simplified implementation)"""
        # This would use a pre-trained model or template matching
        # For now, return mock data
        return [
            {
                'type': 'standard_pallet',
                'bbox': [100, 100, 220, 180],  # Mock detection
                'confidence': 0.8
            }
        ]
    
    def create_synthetic_annotations(self, image_paths: List[str], 
                                   annotation_strategy: str = "semi_automated") -> bool:
        """Create synthetic annotations using various strategies"""
        try:
            if annotation_strategy == "template_matching":
                return self._create_template_based_annotations(image_paths)
            elif annotation_strategy == "contour_detection":
                return self._create_contour_based_annotations(image_paths)
            elif annotation_strategy == "grid_estimation":
                return self._create_grid_based_annotations(image_paths)
            else:
                logger.warning(f"Unknown annotation strategy: {annotation_strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Annotation creation failed: {e}")
            return False
    
    def _create_template_based_annotations(self, image_paths: List[str]) -> bool:
        """Create annotations using template matching"""
        # Load pallet templates
        template_paths = self._get_pallet_templates()
        
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            annotations = []
            
            for template_path in template_paths:
                template = cv2.imread(template_path, 0)
                if template is None:
                    continue
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Multi-scale template matching
                for scale in [0.8, 1.0, 1.2]:
                    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                    
                    if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                        continue
                    
                    result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.6)
                    
                    for pt in zip(*locations[::-1]):
                        x, y = pt
                        w, h = scaled_template.shape[1], scaled_template.shape[0]
                        
                        # Convert to normalized coordinates
                        img_h, img_w = image.shape[:2]
                        norm_bbox = (x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h)
                        
                        annotations.append(Annotation(
                            class_id=self.class_mapping[AnnotationType.PALLET],
                            bbox=norm_bbox,
                            confidence=result[y, x]
                        ))
            
            # Save annotations
            self._save_yolo_annotations(img_path, annotations)
        
        return True
    
    def _create_contour_based_annotations(self, image_paths: List[str]) -> bool:
        """Create annotations using contour detection for racks and spaces"""
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            annotations = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for rack structures
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_h, img_w = image.shape[:2]
            
            for contour in contours:
                # Filter contours by area and aspect ratio
                area = cv2.contourArea(contour)
                if area < 1000:  # Too small
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Classify based on shape characteristics
                if 1.5 <= aspect_ratio <= 4.0 and area > 5000:  # Likely a rack
                    norm_bbox = (x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h)
                    annotations.append(Annotation(
                        class_id=self.class_mapping[AnnotationType.RACK],
                        bbox=norm_bbox,
                        confidence=0.7
                    ))
                elif 0.8 <= aspect_ratio <= 1.5 and 2000 <= area <= 8000:  # Likely a pallet
                    norm_bbox = (x/img_w, y/img_h, (x+w)/img_w, (y+h)/img_h)
                    
                    # Determine if occupied or empty based on content analysis
                    roi = gray[y:y+h, x:x+w]
                    density = np.mean(roi)
                    
                    if density > 120:  # Higher density suggests occupancy
                        class_id = self.class_mapping[AnnotationType.OCCUPIED_SLOT]
                    else:
                        class_id = self.class_mapping[AnnotationType.EMPTY_SLOT]
                    
                    annotations.append(Annotation(
                        class_id=class_id,
                        bbox=norm_bbox,
                        confidence=0.6
                    ))
            
            self._save_yolo_annotations(img_path, annotations)
        
        return True
    
    def _create_grid_based_annotations(self, image_paths: List[str]) -> bool:
        """Create annotations using grid-based rack estimation"""
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            annotations = []
            img_h, img_w = image.shape[:2]
            
            # Estimate rack layout based on perspective and typical warehouse layouts
            rack_regions = self._estimate_rack_regions(image)
            
            for region in rack_regions:
                x1, y1, x2, y2 = region['bounds']
                
                # Create rack annotation
                norm_bbox = (x1/img_w, y1/img_h, x2/img_w, y2/img_h)
                annotations.append(Annotation(
                    class_id=self.class_mapping[AnnotationType.RACK],
                    bbox=norm_bbox,
                    confidence=region['confidence']
                ))
                
                # Subdivide into pallet slots
                slots = self._subdivide_rack_into_slots(region, typical_pallet_size=(120, 80))
                
                for slot in slots:
                    sx1, sy1, sx2, sy2 = slot['bounds']
                    slot_roi = image[sy1:sy2, sx1:sx2]
                    
                    # Analyze slot occupancy
                    occupancy_score = self._analyze_slot_occupancy(slot_roi)
                    
                    if occupancy_score > 0.6:
                        class_id = self.class_mapping[AnnotationType.OCCUPIED_SLOT]
                    else:
                        class_id = self.class_mapping[AnnotationType.EMPTY_SLOT]
                    
                    norm_bbox = (sx1/img_w, sy1/img_h, sx2/img_w, sy2/img_h)
                    annotations.append(Annotation(
                        class_id=class_id,
                        bbox=norm_bbox,
                        confidence=slot['confidence'] * occupancy_score
                    ))
            
            self._save_yolo_annotations(img_path, annotations)
        
        return True
    
    def _estimate_rack_regions(self, image: np.ndarray) -> List[Dict]:
        """Estimate rack regions using perspective analysis"""
        # Simplified implementation - would use more sophisticated computer vision
        img_h, img_w = image.shape[:2]
        
        # Mock rack regions based on typical warehouse layout
        regions = [
            {
                'bounds': [50, 100, img_w//2-50, img_h//2],
                'confidence': 0.7
            },
            {
                'bounds': [img_w//2+50, 100, img_w-50, img_h//2],
                'confidence': 0.7
            }
        ]
        
        return regions
    
    def _subdivide_rack_into_slots(self, rack_region: Dict, typical_pallet_size: Tuple[int, int]) -> List[Dict]:
        """Subdivide rack region into individual pallet slots"""
        x1, y1, x2, y2 = rack_region['bounds']
        width = x2 - x1
        height = y2 - y1
        
        pallet_w, pallet_h = typical_pallet_size
        
        slots_horizontal = max(1, width // pallet_w)
        slots_vertical = max(1, height // pallet_h)
        
        slots = []
        slot_w = width / slots_horizontal
        slot_h = height / slots_vertical
        
        for i in range(int(slots_horizontal)):
            for j in range(int(slots_vertical)):
                sx1 = x1 + i * slot_w
                sy1 = y1 + j * slot_h
                sx2 = sx1 + slot_w
                sy2 = sy1 + slot_h
                
                slots.append({
                    'bounds': [int(sx1), int(sy1), int(sx2), int(sy2)],
                    'confidence': rack_region['confidence'] * 0.8
                })
        
        return slots
    
    def _analyze_slot_occupancy(self, slot_roi: np.ndarray) -> float:
        """Analyze whether a slot is occupied"""
        if slot_roi.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(slot_roi, cv2.COLOR_BGR2GRAY) if len(slot_roi.shape) == 3 else slot_roi
        
        # Multiple indicators of occupancy
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size
        
        # Combine indicators
        intensity_score = 1.0 - abs(mean_intensity - 127.5) / 127.5
        texture_score = min(std_intensity / 50.0, 1.0)
        edge_score = min(edge_density * 20, 1.0)
        
        occupancy_score = (0.4 * intensity_score + 0.3 * texture_score + 0.3 * edge_score)
        return occupancy_score
    
    def _get_pallet_templates(self) -> List[str]:
        """Get paths to pallet templates"""
        # This would return paths to template images
        # For now, return empty list
        return []
    
    def _save_yolo_annotations(self, image_path: str, annotations: List[Annotation]):
        """Save annotations in YOLO format"""
        label_path = image_path.replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # YOLO format: class_id center_x center_y width height
                x1, y1, x2, y2 = ann.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                f.write(f"{ann.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def _save_calibration_data(self):
        """Save calibration parameters"""
        calibration_file = self.project_root / 'calibration' / 'results' / 'calibration.json'
        with open(calibration_file, 'w') as f:
            json.dump(self.calibration_params, f, indent=2)
    
    def create_enhanced_dataset_config(self) -> str:
        """Create enhanced dataset configuration"""
        config = {
            'path': str(self.project_root / 'data'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_mapping),
            'names': [ann_type.value for ann_type in AnnotationType]
        }
        
        config_path = self.project_root / 'data' / 'warehouse_enhanced.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)
    
    def train_enhanced_model(self, 
                           pretrained_model: str = 'yolov8n.pt',
                           use_data_augmentation: bool = True) -> Optional[YOLO]:
        """Train enhanced warehouse-specific model"""
        try:
            # Create dataset config
            data_yaml = self.create_enhanced_dataset_config()
            
            # Initialize model
            model = YOLO(pretrained_model)
            
            # Enhanced training parameters
            training_args = {
                'data': data_yaml,
                'epochs': self.training_config['epochs'],
                'imgsz': self.training_config['img_size'],
                'batch': self.training_config['batch_size'],
                'patience': self.training_config['patience'],
                'save': True,
                'project': str(self.project_root / 'models'),
                'name': 'warehouse_enhanced',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'lrf': 0.1,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': self.training_config['mosaic'],
                'mixup': self.training_config['mixup'],
                'copy_paste': self.training_config['copy_paste']
            }
            
            logger.info("Starting enhanced model training...")
            results = model.train(**training_args)
            
            # Save the best model
            best_model_path = self.project_root / 'models' / 'final' / 'warehouse_best.pt'
            shutil.copy(
                self.project_root / 'models' / 'warehouse_enhanced' / 'weights' / 'best.pt',
                best_model_path
            )
            
            logger.info(f"Enhanced model training completed. Best model saved to: {best_model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Enhanced training failed: {e}")
            return None

def main():
    """Main function for enhanced model training"""
    print("🚀 Enhanced Warehouse Model Training System")
    print("=" * 60)
    
    trainer = EnhancedWarehouseModelTrainer()
    
    # 1. Check for video files
    video_dir = trainer.project_root / 'video'
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    
    if not video_files:
        print("❌ No video files found. Please add videos to the 'video' directory.")
        return
    
    print(f"📹 Found {len(video_files)} video files")
    
    # 2. Extract diverse, high-quality frames
    print("\n📸 Extracting high-quality training frames...")
    all_frames = []
    for video_file in video_files:
        frames = trainer.extract_diverse_frames(str(video_file), frame_count=200)
        all_frames.extend(frames)
    
    print(f"✅ Extracted {len(all_frames)} total frames")
    
    # 3. Setup camera calibration (if reference images available)
    print("\n📏 Setting up camera calibration...")
    reference_images = list((trainer.project_root / 'calibration' / 'reference_images').glob('*.jpg'))
    if reference_images:
        known_dims = {
            'standard_pallet': {'width': 1.2, 'height': 0.8},  # meters
            'euro_pallet': {'width': 1.2, 'height': 0.8}
        }
        trainer.setup_camera_calibration([str(img) for img in reference_images], known_dims)
    
    # 4. Create synthetic annotations
    print("\n🏷️ Creating synthetic annotations...")
    trainer.create_synthetic_annotations(all_frames[:50], "contour_detection")  # Start with subset
    
    # 5. Train enhanced model
    print("\n🧠 Training enhanced warehouse model...")
    model = trainer.train_enhanced_model(use_data_augmentation=True)
    
    if model:
        print("✅ Enhanced model training completed successfully!")
        print(f"📁 Model files saved in: {trainer.project_root / 'models' / 'final'}")
        print("\n📋 Next steps:")
        print("1. Review and refine the synthetic annotations")
        print("2. Add manual annotations for better accuracy")
        print("3. Test the model on validation data")
        print("4. Deploy the enhanced model in your main pipeline")
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    main()
