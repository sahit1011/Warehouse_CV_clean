#!/usr/bin/env python3
"""
space_measurement.py
Enhanced space measurement algorithms for warehouse analysis.
Handles complex scenarios including partial occupancy, obstructions, and irregular shapes.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SpaceType(Enum):
    """Types of storage spaces"""
    PALLET_SLOT = "pallet_slot"
    SHELF_SPACE = "shelf_space"
    FLOOR_AREA = "floor_area"
    OVERHEAD_SPACE = "overhead_space"

class OccupancyLevel(Enum):
    """Levels of space occupancy"""
    EMPTY = "empty"
    PARTIALLY_OCCUPIED = "partially_occupied"
    FULLY_OCCUPIED = "fully_occupied"
    OVERLOADED = "overloaded"

@dataclass
class SpaceMeasurement:
    """Detailed space measurement result"""
    space_id: str
    space_type: SpaceType
    total_area_sqm: float
    usable_area_sqm: float
    occupied_area_sqm: float
    free_area_sqm: float
    occupancy_level: OccupancyLevel
    occupancy_percentage: float
    capacity_units: int
    occupied_units: int
    free_units: int
    obstructions: List[Dict]
    accessibility_score: float  # 0-1, where 1 is fully accessible
    measurement_confidence: float
    notes: str

class EnhancedSpaceMeasurement:
    """Enhanced space measurement system with robust algorithms"""
    
    def __init__(self, pixels_per_meter: float = 100.0):
        self.pixels_per_meter = pixels_per_meter
        self.standard_pallet_size = (1.2, 0.8)  # meters (length, width)
        self.min_pallet_area = 0.5  # minimum area for a pallet slot in sqm
        self.max_pallet_area = 3.0  # maximum area for a pallet slot in sqm

        # Measurement thresholds
        # Special case handling parameters
        self.damaged_rack_indicators = {
            'min_structural_confidence': 0.4,
            'max_deformation_ratio': 0.2,
            'min_edge_continuity': 0.6
        }
        
        self.irregular_stacking_thresholds = {
            'max_height_variation': 0.3,  # 30% height variation tolerance
            'min_alignment_score': 0.5,
            'max_overhang_ratio': 0.15
        }
        
        self.seasonal_adjustment_factors = {
            'summer': 1.0,    # Standard capacity
            'winter': 0.85,   # Reduced for heating equipment
            'holiday': 0.7,   # Reduced for seasonal inventory
            'maintenance': 0.5  # Significantly reduced during maintenance
        }
        self.partial_occupancy_threshold = 0.3  # 30% occupied = partially occupied
        self.full_occupancy_threshold = 0.9     # 90% occupied = fully occupied
        self.overload_threshold = 1.1           # 110% = overloaded
        
    def measure_rack_space(self, rack_bbox: Tuple[int, int, int, int], 
                          pallet_detections: List[Dict], 
                          obstruction_detections: List[Dict],
                          frame: np.ndarray,
                          seasonal_mode: str = "summer",
                          damage_assessment: bool = True) -> SpaceMeasurement:
        """Comprehensive rack space measurement"""
        try:
            # Calculate basic dimensions
            x1, y1, x2, y2 = rack_bbox
            total_area_pixels = (x2 - x1) * (y2 - y1)
            total_area_sqm = self.pixels_to_sqm(total_area_pixels)
            # Check for special cases first
            special_cases = self._detect_special_cases(rack_bbox, pallet_detections, obstruction_detections, frame)
            
            # Apply seasonal adjustments
            seasonal_factor = self.seasonal_adjustment_factors.get(seasonal_mode, 1.0)
            
            # Analyze rack structure and capacity
            capacity_analysis = self.analyze_rack_capacity(rack_bbox, frame)
            
            # Calculate occupied space
            occupied_analysis = self.calculate_occupied_space(
                rack_bbox, pallet_detections, frame
            )
            
            # Assess obstructions
            obstruction_analysis = self.assess_obstructions(
                rack_bbox, obstruction_detections, frame
            )
            
            # Calculate usable area (total - permanent obstructions)
            usable_area_sqm = total_area_sqm - obstruction_analysis['permanent_obstruction_area']
            
            # Calculate free space
            free_area_sqm = max(0, usable_area_sqm - occupied_analysis['occupied_area_sqm'])
            
            # Determine occupancy level
            occupancy_percentage = (occupied_analysis['occupied_area_sqm'] / usable_area_sqm * 100) if usable_area_sqm > 0 else 0
            occupancy_level = self.determine_occupancy_level(occupancy_percentage)
            
            # Calculate accessibility score
            accessibility_score = self.calculate_accessibility_score(
                rack_bbox, obstruction_analysis, frame
            )
            
            # Generate space ID
            space_id = self.generate_space_id(rack_bbox, frame.shape)
            
            # Compile measurement notes
            notes = self.generate_measurement_notes(
                capacity_analysis, occupied_analysis, obstruction_analysis, accessibility_score
            )
            
            return SpaceMeasurement(
                space_id=space_id,
                space_type=SpaceType.PALLET_SLOT,
                total_area_sqm=total_area_sqm,
                usable_area_sqm=usable_area_sqm,
                occupied_area_sqm=occupied_analysis['occupied_area_sqm'],
                free_area_sqm=free_area_sqm,
                occupancy_level=occupancy_level,
                occupancy_percentage=occupancy_percentage,
                capacity_units=capacity_analysis['total_slots'],
                occupied_units=occupied_analysis['occupied_slots'],
                free_units=max(0, capacity_analysis['total_slots'] - occupied_analysis['occupied_slots']),
                obstructions=obstruction_analysis['obstructions'],
                accessibility_score=accessibility_score,
                measurement_confidence=self.calculate_measurement_confidence(
                    capacity_analysis, occupied_analysis, obstruction_analysis
                ),
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error measuring rack space: {e}")
            return self.create_fallback_measurement(rack_bbox)
    
    def analyze_rack_capacity(self, rack_bbox: Tuple[int, int, int, int], 
                             frame: np.ndarray) -> Dict:
        """Analyze rack structure to determine capacity"""
        x1, y1, x2, y2 = rack_bbox
        rack_width = x2 - x1
        rack_height = y2 - y1
        
        # Estimate pallet slots based on rack dimensions
        pallet_width_pixels = self.standard_pallet_size[0] * self.pixels_per_meter
        pallet_height_pixels = self.standard_pallet_size[1] * self.pixels_per_meter
        
        # Calculate potential slots
        horizontal_slots = max(1, int(rack_width / pallet_width_pixels))
        vertical_slots = max(1, int(rack_height / pallet_height_pixels))
        
        # Adjust for rack type (most warehouse racks are horizontal)
        if rack_width > rack_height * 1.5:  # Wide rack
            total_slots = horizontal_slots
            rack_type = "horizontal"
        else:  # Tall rack or square
            total_slots = horizontal_slots * min(vertical_slots, 3)  # Max 3 levels
            rack_type = "multi-level"
        
        return {
            'total_slots': total_slots,
            'horizontal_slots': horizontal_slots,
            'vertical_slots': vertical_slots,
            'rack_type': rack_type,
            'estimated_slot_size': (pallet_width_pixels, pallet_height_pixels)
        }
    
    def calculate_occupied_space(self, rack_bbox: Tuple[int, int, int, int], 
                               pallet_detections: List[Dict], 
                               frame: np.ndarray) -> Dict:
        """Calculate occupied space within rack"""
        rx1, ry1, rx2, ry2 = rack_bbox
        
        occupied_area_pixels = 0
        occupied_slots = 0
        pallet_positions = []
        
        for pallet in pallet_detections:
            px1, py1, px2, py2 = pallet['bbox']
            
            # Check if pallet overlaps with rack
            overlap_x1 = max(rx1, px1)
            overlap_y1 = max(ry1, py1)
            overlap_x2 = min(rx2, px2)
            overlap_y2 = min(ry2, py2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # Calculate overlap area
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                pallet_area = (px2 - px1) * (py2 - py1)
                
                # If significant overlap (>50% of pallet), count as occupied
                if overlap_area > pallet_area * 0.5:
                    occupied_area_pixels += overlap_area
                    occupied_slots += 1
                    pallet_positions.append({
                        'bbox': (px1, py1, px2, py2),
                        'overlap_area': overlap_area,
                        'confidence': pallet.get('confidence', 0.5)
                    })
        
        return {
            'occupied_area_sqm': self.pixels_to_sqm(occupied_area_pixels),
            'occupied_slots': occupied_slots,
            'pallet_positions': pallet_positions
        }
    
    def assess_obstructions(self, rack_bbox: Tuple[int, int, int, int], 
                          obstruction_detections: List[Dict], 
                          frame: np.ndarray) -> Dict:
        """Assess obstructions and their impact on usable space"""
        rx1, ry1, rx2, ry2 = rack_bbox
        
        obstructions = []
        total_obstruction_area = 0
        permanent_obstruction_area = 0
        
        for obstruction in obstruction_detections:
            ox1, oy1, ox2, oy2 = obstruction['bbox']
            
            # Check if obstruction overlaps with rack
            overlap_x1 = max(rx1, ox1)
            overlap_y1 = max(ry1, oy1)
            overlap_x2 = min(rx2, ox2)
            overlap_y2 = min(ry2, oy2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                
                # Classify obstruction type
                obstruction_type = self.classify_obstruction(obstruction, frame)
                
                obstruction_info = {
                    'bbox': (ox1, oy1, ox2, oy2),
                    'overlap_area_pixels': overlap_area,
                    'overlap_area_sqm': self.pixels_to_sqm(overlap_area),
                    'type': obstruction_type,
                    'confidence': obstruction.get('confidence', 0.5)
                }
                
                obstructions.append(obstruction_info)
                total_obstruction_area += overlap_area
                
                # Permanent obstructions reduce usable space
                if obstruction_type in ['structural', 'equipment']:
                    permanent_obstruction_area += overlap_area
        
        return {
            'obstructions': obstructions,
            'total_obstruction_area': self.pixels_to_sqm(total_obstruction_area),
            'permanent_obstruction_area': self.pixels_to_sqm(permanent_obstruction_area),
            'temporary_obstruction_area': self.pixels_to_sqm(total_obstruction_area - permanent_obstruction_area)
        }
    
    def classify_obstruction(self, obstruction: Dict, frame: np.ndarray) -> str:
        """Classify type of obstruction"""
        # This could be enhanced with additional ML models
        # For now, use simple heuristics
        
        ox1, oy1, ox2, oy2 = obstruction['bbox']
        width = ox2 - ox1
        height = oy2 - oy1
        aspect_ratio = width / height if height > 0 else 1
        
        if aspect_ratio > 3:  # Very wide
            return 'equipment'
        elif aspect_ratio < 0.5:  # Very tall
            return 'structural'
        else:
            return 'temporary'
    
    def determine_occupancy_level(self, occupancy_percentage: float) -> OccupancyLevel:
        """Determine occupancy level based on percentage"""
        if occupancy_percentage >= self.overload_threshold * 100:
            return OccupancyLevel.OVERLOADED
        elif occupancy_percentage >= self.full_occupancy_threshold * 100:
            return OccupancyLevel.FULLY_OCCUPIED
        elif occupancy_percentage >= self.partial_occupancy_threshold * 100:
            return OccupancyLevel.PARTIALLY_OCCUPIED
        else:
            return OccupancyLevel.EMPTY
    
    def calculate_accessibility_score(self, rack_bbox: Tuple[int, int, int, int], 
                                    obstruction_analysis: Dict, 
                                    frame: np.ndarray) -> float:
        """Calculate accessibility score (0-1)"""
        base_score = 1.0
        
        # Reduce score based on obstructions
        for obstruction in obstruction_analysis['obstructions']:
            if obstruction['type'] == 'structural':
                base_score -= 0.3
            elif obstruction['type'] == 'equipment':
                base_score -= 0.2
            else:  # temporary
                base_score -= 0.1
        
        # Consider position (edge racks might be less accessible)
        x1, y1, x2, y2 = rack_bbox
        frame_height, frame_width = frame.shape[:2]
        center_x = (x1 + x2) / 2
        
        # Reduce score for racks at edges
        if center_x < frame_width * 0.1 or center_x > frame_width * 0.9:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def calculate_measurement_confidence(self, capacity_analysis: Dict, 
                                       occupied_analysis: Dict, 
                                       obstruction_analysis: Dict) -> float:
        """Calculate overall measurement confidence"""
        base_confidence = 0.8
        
        # Reduce confidence based on various factors
        if len(occupied_analysis['pallet_positions']) == 0:
            base_confidence += 0.1  # Higher confidence when no pallets detected
        
        # Reduce confidence for complex obstruction scenarios
        if len(obstruction_analysis['obstructions']) > 2:
            base_confidence -= 0.2
        
        # Adjust based on detection confidence
        avg_pallet_confidence = np.mean([p['confidence'] for p in occupied_analysis['pallet_positions']]) if occupied_analysis['pallet_positions'] else 1.0
        base_confidence = (base_confidence + avg_pallet_confidence) / 2
        
        return max(0.1, min(1.0, base_confidence))
    
    def generate_measurement_notes(self, capacity_analysis: Dict, 
                                 occupied_analysis: Dict, 
                                 obstruction_analysis: Dict, 
                                 accessibility_score: float) -> str:
        """Generate descriptive notes for the measurement"""
        notes = []
        
        # Capacity notes
        notes.append(f"{capacity_analysis['rack_type']} rack with {capacity_analysis['total_slots']} slots")
        
        # Occupancy notes
        if occupied_analysis['occupied_slots'] > 0:
            notes.append(f"{occupied_analysis['occupied_slots']} slots occupied")
        
        # Obstruction notes
        if obstruction_analysis['obstructions']:
            obstruction_types = [o['type'] for o in obstruction_analysis['obstructions']]
            notes.append(f"Obstructions: {', '.join(set(obstruction_types))}")
        
        # Accessibility notes
        if accessibility_score < 0.7:
            notes.append("Limited accessibility")
        
        return "; ".join(notes)
    
    def create_fallback_measurement(self, rack_bbox: Tuple[int, int, int, int]) -> SpaceMeasurement:
        """Create fallback measurement when analysis fails"""
        x1, y1, x2, y2 = rack_bbox
        total_area_sqm = self.pixels_to_sqm((x2 - x1) * (y2 - y1))
        
        return SpaceMeasurement(
            space_id=f"FALLBACK_{x1}_{y1}",
            space_type=SpaceType.PALLET_SLOT,
            total_area_sqm=total_area_sqm,
            usable_area_sqm=total_area_sqm,
            occupied_area_sqm=0.0,
            free_area_sqm=total_area_sqm,
            occupancy_level=OccupancyLevel.EMPTY,
            occupancy_percentage=0.0,
            capacity_units=1,
            occupied_units=0,
            free_units=1,
            obstructions=[],
            accessibility_score=0.5,
            measurement_confidence=0.3,
            notes="Fallback measurement - analysis failed"
        )
    
    def pixels_to_sqm(self, pixels: float) -> float:
        """Convert pixel area to square meters"""
        return pixels / (self.pixels_per_meter ** 2)
    
    def _detect_special_cases(self, rack_bbox: Tuple[int, int, int, int], 
                             pallet_detections: List[Dict], 
                             obstruction_detections: List[Dict], 
                             frame: np.ndarray) -> Dict:
        """Detect special cases that need custom handling"""
        special_cases = {
            'damaged_rack': False,
            'irregular_stacking': False,
            'seasonal_adjustment_needed': False,
            'partial_visibility': False
        }
        
        # Check for damaged rack indicators
        x1, y1, x2, y2 = rack_bbox
        rack_area = (x2 - x1) * (y2 - y1)
        
        # Simple heuristics for special case detection
        if len(obstruction_detections) > 3:
            special_cases['damaged_rack'] = True
        
        if len(pallet_detections) > 0:
            # Check for irregular stacking patterns
            pallet_heights = [p['bbox'][3] - p['bbox'][1] for p in pallet_detections]
            if len(set(pallet_heights)) > 2:  # Multiple different heights
                special_cases['irregular_stacking'] = True
        
        return special_cases
    
    def generate_space_id(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> str:
        """Generate unique space identifier"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        frame_height, frame_width = frame_shape[:2]
        
        # Grid-based ID
        grid_x = int((center_x / frame_width) * 10)  # 10x10 grid
        grid_y = int((center_y / frame_height) * 10)
        
        return f"SPACE_{grid_x:02d}_{grid_y:02d}"
