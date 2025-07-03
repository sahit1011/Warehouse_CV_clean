#!/usr/bin/env python3
"""
exception_handler.py
Comprehensive exception handling and edge case management for warehouse analysis.
Handles damaged racks, blocked areas, irregular shapes, and measurement uncertainties.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExceptionType(Enum):
    """Types of exceptions that can occur during analysis"""
    DAMAGED_RACK = "damaged_rack"
    BLOCKED_ACCESS = "blocked_access"
    IRREGULAR_SHAPE = "irregular_shape"
    POOR_LIGHTING = "poor_lighting"
    MEASUREMENT_UNCERTAINTY = "measurement_uncertainty"
    DETECTION_FAILURE = "detection_failure"
    CALIBRATION_ERROR = "calibration_error"
    OBSTRUCTION_OVERLAP = "obstruction_overlap"

class SeverityLevel(Enum):
    """Severity levels for exceptions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ExceptionReport:
    """Detailed exception report"""
    exception_type: ExceptionType
    severity: SeverityLevel
    description: str
    affected_area: Tuple[int, int, int, int]  # bbox
    confidence: float
    suggested_action: str
    fallback_measurement: Optional[Dict]
    metadata: Dict[str, Any]

class WarehouseExceptionHandler:
    """Comprehensive exception handling system for warehouse analysis"""
    
    def __init__(self):
        self.exception_history = []
        self.severity_thresholds = {
            'confidence': 0.5,
            'lighting_variance': 50,
            'shape_irregularity': 0.3,
            'measurement_uncertainty': 0.2
        }
        
    def handle_analysis_exceptions(self, frame: np.ndarray, 
                                 detections: List[Dict], 
                                 analysis_results: List[Dict]) -> Tuple[List[Dict], List[ExceptionReport]]:
        """Main exception handling pipeline"""
        exceptions = []
        corrected_results = []
        
        try:
            # Check for lighting issues
            lighting_exceptions = self.check_lighting_conditions(frame)
            exceptions.extend(lighting_exceptions)
            
            # Check for detection failures
            detection_exceptions = self.check_detection_quality(detections, frame)
            exceptions.extend(detection_exceptions)
            
            # Process each analysis result
            for result in analysis_results:
                try:
                    # Check for damaged racks
                    damage_exceptions = self.check_rack_damage(result, frame)
                    exceptions.extend(damage_exceptions)
                    
                    # Check for blocked access
                    access_exceptions = self.check_access_blockage(result, frame)
                    exceptions.extend(access_exceptions)
                    
                    # Check for irregular shapes
                    shape_exceptions = self.check_shape_irregularities(result, frame)
                    exceptions.extend(shape_exceptions)
                    
                    # Check measurement uncertainty
                    uncertainty_exceptions = self.check_measurement_uncertainty(result)
                    exceptions.extend(uncertainty_exceptions)
                    
                    # Apply corrections and fallbacks
                    corrected_result = self.apply_corrections(result, exceptions)
                    corrected_results.append(corrected_result)
                    
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    # Create critical exception
                    critical_exception = ExceptionReport(
                        exception_type=ExceptionType.DETECTION_FAILURE,
                        severity=SeverityLevel.CRITICAL,
                        description=f"Critical analysis failure: {str(e)}",
                        affected_area=result.get('bbox', (0, 0, 100, 100)),
                        confidence=0.0,
                        suggested_action="Manual inspection required",
                        fallback_measurement=self.create_emergency_fallback(result),
                        metadata={'error': str(e)}
                    )
                    exceptions.append(critical_exception)
                    corrected_results.append(self.create_emergency_fallback(result))
            
            # Log exceptions
            self.log_exceptions(exceptions)
            
            return corrected_results, exceptions
            
        except Exception as e:
            logger.critical(f"Critical failure in exception handling: {e}")
            return analysis_results, []  # Return original results if exception handling fails
    
    def check_lighting_conditions(self, frame: np.ndarray) -> List[ExceptionReport]:
        """Check for poor lighting conditions"""
        exceptions = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate lighting metrics
            mean_brightness = np.mean(gray)
            brightness_variance = np.var(gray)
            
            # Check for poor lighting
            if mean_brightness < 50:  # Too dark
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.POOR_LIGHTING,
                    severity=SeverityLevel.HIGH,
                    description=f"Poor lighting detected - mean brightness: {mean_brightness:.1f}",
                    affected_area=(0, 0, frame.shape[1], frame.shape[0]),
                    confidence=0.8,
                    suggested_action="Improve lighting or adjust camera settings",
                    fallback_measurement=None,
                    metadata={'mean_brightness': mean_brightness, 'variance': brightness_variance}
                ))
            elif mean_brightness > 200:  # Too bright/overexposed
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.POOR_LIGHTING,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Overexposure detected - mean brightness: {mean_brightness:.1f}",
                    affected_area=(0, 0, frame.shape[1], frame.shape[0]),
                    confidence=0.7,
                    suggested_action="Reduce exposure or adjust camera settings",
                    fallback_measurement=None,
                    metadata={'mean_brightness': mean_brightness, 'variance': brightness_variance}
                ))
            
            # Check for uneven lighting
            if brightness_variance > self.severity_thresholds['lighting_variance']:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.POOR_LIGHTING,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Uneven lighting detected - variance: {brightness_variance:.1f}",
                    affected_area=(0, 0, frame.shape[1], frame.shape[0]),
                    confidence=0.6,
                    suggested_action="Improve lighting uniformity",
                    fallback_measurement=None,
                    metadata={'mean_brightness': mean_brightness, 'variance': brightness_variance}
                ))
                
        except Exception as e:
            logger.error(f"Error checking lighting conditions: {e}")
            
        return exceptions
    
    def check_detection_quality(self, detections: List[Dict], frame: np.ndarray) -> List[ExceptionReport]:
        """Check for detection quality issues"""
        exceptions = []
        
        try:
            # Check for too few detections
            if len(detections) == 0:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.DETECTION_FAILURE,
                    severity=SeverityLevel.HIGH,
                    description="No objects detected in frame",
                    affected_area=(0, 0, frame.shape[1], frame.shape[0]),
                    confidence=0.9,
                    suggested_action="Check model performance or retrain",
                    fallback_measurement=None,
                    metadata={'detection_count': 0}
                ))
            
            # Check for low confidence detections
            low_confidence_count = sum(1 for d in detections if d.get('confidence', 1.0) < self.severity_thresholds['confidence'])
            if low_confidence_count > len(detections) * 0.5:  # More than 50% low confidence
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.DETECTION_FAILURE,
                    severity=SeverityLevel.MEDIUM,
                    description=f"{low_confidence_count} low confidence detections",
                    affected_area=(0, 0, frame.shape[1], frame.shape[0]),
                    confidence=0.7,
                    suggested_action="Review detection thresholds or model quality",
                    fallback_measurement=None,
                    metadata={'low_confidence_count': low_confidence_count, 'total_detections': len(detections)}
                ))
                
        except Exception as e:
            logger.error(f"Error checking detection quality: {e}")
            
        return exceptions
    
    def check_rack_damage(self, result: Dict, frame: np.ndarray) -> List[ExceptionReport]:
        """Check for rack damage indicators"""
        exceptions = []
        
        try:
            # Check confidence levels
            if result.get('confidence', 1.0) < 0.4:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.DAMAGED_RACK,
                    severity=SeverityLevel.HIGH,
                    description=f"Possible rack damage - low detection confidence: {result.get('confidence', 0):.2f}",
                    affected_area=result.get('bbox', (0, 0, 100, 100)),
                    confidence=0.6,
                    suggested_action="Manual inspection required",
                    fallback_measurement=self.create_damage_fallback(result),
                    metadata={'detection_confidence': result.get('confidence', 0)}
                ))
            
            # Check measurement confidence
            if result.get('measurement_confidence', 1.0) < 0.3:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.DAMAGED_RACK,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Measurement uncertainty suggests possible damage",
                    affected_area=result.get('bbox', (0, 0, 100, 100)),
                    confidence=0.5,
                    suggested_action="Verify rack structural integrity",
                    fallback_measurement=None,
                    metadata={'measurement_confidence': result.get('measurement_confidence', 0)}
                ))
                
        except Exception as e:
            logger.error(f"Error checking rack damage: {e}")
            
        return exceptions
    
    def check_access_blockage(self, result: Dict, frame: np.ndarray) -> List[ExceptionReport]:
        """Check for access blockage issues"""
        exceptions = []
        
        try:
            accessibility_score = result.get('accessibility_score', 1.0)
            
            if accessibility_score < 0.3:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.BLOCKED_ACCESS,
                    severity=SeverityLevel.HIGH,
                    description=f"Severely blocked access - score: {accessibility_score:.2f}",
                    affected_area=result.get('bbox', (0, 0, 100, 100)),
                    confidence=0.8,
                    suggested_action="Clear obstructions or mark as inaccessible",
                    fallback_measurement=self.create_blocked_fallback(result),
                    metadata={'accessibility_score': accessibility_score, 'obstructions': result.get('obstructions_count', 0)}
                ))
            elif accessibility_score < 0.6:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.BLOCKED_ACCESS,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Partially blocked access - score: {accessibility_score:.2f}",
                    affected_area=result.get('bbox', (0, 0, 100, 100)),
                    confidence=0.6,
                    suggested_action="Consider clearing minor obstructions",
                    fallback_measurement=None,
                    metadata={'accessibility_score': accessibility_score, 'obstructions': result.get('obstructions_count', 0)}
                ))
                
        except Exception as e:
            logger.error(f"Error checking access blockage: {e}")
            
        return exceptions
    
    def check_shape_irregularities(self, result: Dict, frame: np.ndarray) -> List[ExceptionReport]:
        """Check for irregular rack shapes"""
        exceptions = []
        
        try:
            bbox = result.get('bbox', (0, 0, 100, 100))
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.IRREGULAR_SHAPE,
                    severity=SeverityLevel.HIGH,
                    description="Invalid bounding box dimensions",
                    affected_area=bbox,
                    confidence=0.9,
                    suggested_action="Review detection algorithm",
                    fallback_measurement=self.create_shape_fallback(result),
                    metadata={'width': width, 'height': height}
                ))
            else:
                aspect_ratio = width / height
                # Check for extremely unusual aspect ratios
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    exceptions.append(ExceptionReport(
                        exception_type=ExceptionType.IRREGULAR_SHAPE,
                        severity=SeverityLevel.MEDIUM,
                        description=f"Unusual aspect ratio: {aspect_ratio:.2f}",
                        affected_area=bbox,
                        confidence=0.6,
                        suggested_action="Verify rack dimensions",
                        fallback_measurement=None,
                        metadata={'aspect_ratio': aspect_ratio, 'width': width, 'height': height}
                    ))
                    
        except Exception as e:
            logger.error(f"Error checking shape irregularities: {e}")
            
        return exceptions
    
    def check_measurement_uncertainty(self, result: Dict) -> List[ExceptionReport]:
        """Check for measurement uncertainty issues"""
        exceptions = []
        
        try:
            measurement_confidence = result.get('measurement_confidence', 1.0)
            
            if measurement_confidence < self.severity_thresholds['measurement_uncertainty']:
                exceptions.append(ExceptionReport(
                    exception_type=ExceptionType.MEASUREMENT_UNCERTAINTY,
                    severity=SeverityLevel.MEDIUM,
                    description=f"High measurement uncertainty: {1-measurement_confidence:.2f}",
                    affected_area=result.get('bbox', (0, 0, 100, 100)),
                    confidence=measurement_confidence,
                    suggested_action="Consider manual verification",
                    fallback_measurement=None,
                    metadata={'measurement_confidence': measurement_confidence}
                ))
                
        except Exception as e:
            logger.error(f"Error checking measurement uncertainty: {e}")
            
        return exceptions
    
    def apply_corrections(self, result: Dict, exceptions: List[ExceptionReport]) -> Dict:
        """Apply corrections based on detected exceptions"""
        corrected_result = result.copy()
        
        try:
            # Apply corrections based on exception types
            for exception in exceptions:
                if exception.affected_area == result.get('bbox', (0, 0, 100, 100)):
                    if exception.exception_type == ExceptionType.DAMAGED_RACK:
                        corrected_result['condition'] = 'damaged'
                        corrected_result['notes'] = f"{corrected_result.get('notes', '')}; {exception.description}"
                    
                    elif exception.exception_type == ExceptionType.BLOCKED_ACCESS:
                        if exception.severity == SeverityLevel.HIGH:
                            corrected_result['condition'] = 'inaccessible'
                        else:
                            corrected_result['condition'] = 'partially-blocked'
                        corrected_result['notes'] = f"{corrected_result.get('notes', '')}; {exception.description}"
                    
                    elif exception.exception_type == ExceptionType.MEASUREMENT_UNCERTAINTY:
                        # Reduce confidence in measurements
                        corrected_result['confidence'] = min(corrected_result.get('confidence', 1.0), 0.5)
                        corrected_result['notes'] = f"{corrected_result.get('notes', '')}; High uncertainty"
                    
                    # Apply fallback measurements if available
                    if exception.fallback_measurement:
                        for key, value in exception.fallback_measurement.items():
                            corrected_result[key] = value
                            
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            
        return corrected_result
    
    def create_damage_fallback(self, result: Dict) -> Dict:
        """Create fallback measurement for damaged racks"""
        return {
            'condition': 'damaged',
            'free_space': 0,
            'accessibility_score': 0.1,
            'notes': 'Damaged rack - manual inspection required'
        }
    
    def create_blocked_fallback(self, result: Dict) -> Dict:
        """Create fallback measurement for blocked racks"""
        return {
            'condition': 'inaccessible',
            'free_space': 0,
            'accessibility_score': 0.0,
            'notes': 'Access blocked - clear obstructions'
        }
    
    def create_shape_fallback(self, result: Dict) -> Dict:
        """Create fallback measurement for irregular shapes"""
        return {
            'total_capacity': 1,
            'occupied_space': 0,
            'free_space': 1,
            'notes': 'Irregular shape detected - verify dimensions'
        }
    
    def create_emergency_fallback(self, result: Dict) -> Dict:
        """Create emergency fallback when all else fails"""
        return {
            'aisle': 'UNKNOWN',
            'rack_id': 'ERROR',
            'rack_type': 'unknown',
            'total_capacity': 0,
            'occupied_space': 0,
            'free_space': 0,
            'condition': 'error',
            'confidence': 0.0,
            'bbox': result.get('bbox', (0, 0, 100, 100)),
            'area_sqm': 0.0,
            'occupancy_percentage': 0.0,
            'notes': 'Analysis failed - manual inspection required'
        }
    
    def log_exceptions(self, exceptions: List[ExceptionReport]):
        """Log exceptions for monitoring and analysis"""
        for exception in exceptions:
            self.exception_history.append(exception)
            
            if exception.severity == SeverityLevel.CRITICAL:
                logger.critical(f"CRITICAL: {exception.description}")
            elif exception.severity == SeverityLevel.HIGH:
                logger.error(f"HIGH: {exception.description}")
            elif exception.severity == SeverityLevel.MEDIUM:
                logger.warning(f"MEDIUM: {exception.description}")
            else:
                logger.info(f"LOW: {exception.description}")
    
    def get_exception_summary(self) -> Dict:
        """Get summary of all exceptions encountered"""
        if not self.exception_history:
            return {'total': 0, 'by_type': {}, 'by_severity': {}}
        
        by_type = {}
        by_severity = {}
        
        for exception in self.exception_history:
            # Count by type
            type_name = exception.exception_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            
            # Count by severity
            severity_name = exception.severity.value
            by_severity[severity_name] = by_severity.get(severity_name, 0) + 1
        
        return {
            'total': len(self.exception_history),
            'by_type': by_type,
            'by_severity': by_severity
        }
