#!/usr/bin/env python3
"""
visualization_tool.py
Interactive visualization tool for warehouse rack detection and space measurement.
Shows bounding boxes, predictions, and empty space analysis with confidence scores.
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import argparse

class WarehouseVisualizationTool:
    def __init__(self, model_path="models/warehouse_yolov8.pt", confidence_threshold=0.3):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.colors = {
            'empty_slot': (0, 255, 0),      # Green for empty spaces
            'occupied_slot': (0, 0, 255),   # Red for occupied spaces  
            'rack': (255, 255, 0),          # Yellow for rack structure
            'pallet': (255, 0, 255),        # Magenta for pallets
            'obstruction': (0, 165, 255),   # Orange for obstructions
            'person': (128, 0, 128),        # Purple for people
            'vehicle': (255, 165, 0),       # Orange for vehicles
        }
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load YOLO model for detection"""
        try:
            if os.path.exists(self.model_path):
                print(f"🔄 Loading enhanced model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print("⚠️  Enhanced model not found, using pre-trained YOLOv8n")
                self.model = YOLO('yolov8n.pt')
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_class_name(self, class_id):
        """Map class ID to warehouse-specific name"""
        class_map = {
            0: 'empty_slot',     # Our trained classes
            1: 'occupied_slot',
            2: 'rack',
            
            # Standard COCO classes that might be relevant
            0: 'person',         # COCO class 0
            2: 'car',           # COCO class 2  
            7: 'truck',         # COCO class 7
            24: 'backpack',     # COCO class 24 (could be cargo)
            28: 'suitcase',     # COCO class 28 (could be boxes)
            39: 'bottle',       # COCO class 39
            41: 'cup',          # COCO class 41
            67: 'dining table', # COCO class 67 (could be shelves)
            72: 'tv',           # COCO class 72
            84: 'book',         # COCO class 84
        }
        return class_map.get(class_id, f'object_{class_id}')
    
    def detect_objects(self, frame):
        """Run object detection on frame"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame)
            detections = []
            
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            'bbox': tuple(bbox),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.get_class_name(class_id)
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def simulate_warehouse_racks(self, frame, detections):
        """Simulate warehouse rack detection for demonstration"""
        height, width = frame.shape[:2]
        simulated_racks = []
        
        # If we have actual detections, enhance them as potential racks
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            
            # Consider larger objects as potential racks
            if area > 5000:  # Minimum area threshold
                # Determine if it's more likely empty or occupied
                # For demo, randomly assign but bias toward empty spaces
                is_empty = np.random.random() > 0.4  # 60% chance of being empty
                capacity = np.random.randint(1, 4)
                occupied = 0 if is_empty else np.random.randint(1, capacity + 1)
                
                simulated_racks.append({
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'rack_id': f"R-{np.random.randint(100, 999)}",
                    'aisle': f"A-{np.random.randint(1, 6)}",
                    'total_capacity': capacity,
                    'occupied_space': occupied,
                    'free_space': capacity - occupied,
                    'condition': np.random.choice(['good', 'partially-blocked', 'damaged'], p=[0.8, 0.15, 0.05]),
                    'empty_percentage': ((capacity - occupied) / capacity * 100) if capacity > 0 else 0
                })
        
        # Add some additional simulated racks if we don't have enough detections
        while len(simulated_racks) < 3:
            # Generate random but realistic rack positions
            rack_width = np.random.randint(100, 300)
            rack_height = np.random.randint(80, 200)
            x1 = np.random.randint(50, width - rack_width - 50)
            y1 = np.random.randint(50, height - rack_height - 50)
            x2 = x1 + rack_width
            y2 = y1 + rack_height
            
            capacity = np.random.randint(1, 4)
            occupied = np.random.randint(0, capacity + 1)
            
            simulated_racks.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': np.random.uniform(0.6, 0.9),
                'rack_id': f"R-{np.random.randint(100, 999)}",
                'aisle': f"A-{np.random.randint(1, 6)}",
                'total_capacity': capacity,
                'occupied_space': occupied,
                'free_space': capacity - occupied,
                'condition': np.random.choice(['good', 'partially-blocked', 'damaged'], p=[0.8, 0.15, 0.05]),
                'empty_percentage': ((capacity - occupied) / capacity * 100) if capacity > 0 else 0
            })
        
        return simulated_racks
    
    def draw_detections(self, frame, detections, simulated_racks):
        """Draw detection results on frame"""
        annotated_frame = frame.copy()
        
        # Draw raw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Choose color based on class
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw simulated warehouse racks with enhanced visualization
        for rack in simulated_racks:
            x1, y1, x2, y2 = rack['bbox']
            
            # Color based on empty space percentage
            empty_pct = rack['empty_percentage']
            if empty_pct > 70:
                rack_color = (0, 255, 0)      # Green for mostly empty
            elif empty_pct > 30:
                rack_color = (0, 255, 255)    # Yellow for partially full
            else:
                rack_color = (0, 0, 255)      # Red for mostly full
            
            # Draw thick rack border
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), rack_color, 3)
            
            # Draw capacity visualization inside rack
            rack_width = x2 - x1
            rack_height = y2 - y1
            
            # Draw capacity slots
            slot_width = rack_width // max(rack['total_capacity'], 1)
            for i in range(rack['total_capacity']):
                slot_x1 = x1 + i * slot_width
                slot_x2 = slot_x1 + slot_width - 2
                slot_y1 = y1 + 5
                slot_y2 = y2 - 5
                
                # Color based on occupancy
                if i < rack['occupied_space']:
                    slot_color = (0, 0, 255)  # Red for occupied
                else:
                    slot_color = (0, 255, 0)  # Green for empty
                
                cv2.rectangle(annotated_frame, (slot_x1, slot_y1), (slot_x2, slot_y2), slot_color, -1)
                cv2.rectangle(annotated_frame, (slot_x1, slot_y1), (slot_x2, slot_y2), (255, 255, 255), 1)
            
            # Add detailed label
            info_lines = [
                f"Rack {rack['rack_id']} ({rack['aisle']})",
                f"Free: {rack['free_space']}/{rack['total_capacity']} ({empty_pct:.0f}%)",
                f"Condition: {rack['condition']}",
                f"Conf: {rack['confidence']:.2f}"
            ]
            
            # Draw info box
            line_height = 20
            text_box_height = len(info_lines) * line_height + 10
            text_box_width = 200
            
            # Position text box (try to keep it within frame)
            text_x = max(0, min(x1, annotated_frame.shape[1] - text_box_width))
            text_y = max(text_box_height, y1 - 10)
            
            # Draw background
            cv2.rectangle(annotated_frame, 
                         (text_x, text_y - text_box_height), 
                         (text_x + text_box_width, text_y), 
                         (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, 
                         (text_x, text_y - text_box_height), 
                         (text_x + text_box_width, text_y), 
                         rack_color, 2)
            
            # Draw text
            for i, line in enumerate(info_lines):
                cv2.putText(annotated_frame, line, 
                           (text_x + 5, text_y - text_box_height + 15 + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_frame
    
    def create_matplotlib_visualization(self, frame, detections, simulated_racks):
        """Create detailed matplotlib visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original frame
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Frame")
        ax1.axis('off')
        
        # Raw detections
        ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                           edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f"{det['class_name']}: {det['confidence']:.2f}", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax2.set_title(f"Raw Detections ({len(detections)} objects)")
        ax2.axis('off')
        
        # Warehouse rack analysis
        ax3.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for rack in simulated_racks:
            x1, y1, x2, y2 = rack['bbox']
            empty_pct = rack['empty_percentage']
            
            # Color based on empty space
            if empty_pct > 70:
                color = 'green'
            elif empty_pct > 30:
                color = 'yellow'
            else:
                color = 'red'
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, 
                           edgecolor=color, facecolor='none')
            ax3.add_patch(rect)
            
            # Add capacity visualization
            slot_width = (x2 - x1) / max(rack['total_capacity'], 1)
            for i in range(rack['total_capacity']):
                slot_x = x1 + i * slot_width
                slot_color = 'red' if i < rack['occupied_space'] else 'green'
                slot_rect = Rectangle((slot_x, y1), slot_width-2, y2-y1-2, 
                                    linewidth=1, edgecolor='white', facecolor=slot_color, alpha=0.6)
                ax3.add_patch(slot_rect)
            
            ax3.text(x1, y1-5, f"{rack['rack_id']}: {rack['free_space']}/{rack['total_capacity']} free", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax3.set_title(f"Warehouse Rack Analysis ({len(simulated_racks)} racks)")
        ax3.axis('off')
        
        # Statistics
        if simulated_racks:
            total_capacity = sum(r['total_capacity'] for r in simulated_racks)
            total_free = sum(r['free_space'] for r in simulated_racks)
            total_occupied = sum(r['occupied_space'] for r in simulated_racks)
            avg_utilization = (total_occupied / total_capacity * 100) if total_capacity > 0 else 0
            
            stats_text = f"""Warehouse Statistics:
            
Total Racks: {len(simulated_racks)}
Total Capacity: {total_capacity} units
Free Space: {total_free} units
Occupied: {total_occupied} units
Utilization: {avg_utilization:.1f}%

Rack Details:"""
            
            for rack in simulated_racks:
                stats_text += f"\n{rack['rack_id']} ({rack['aisle']}): {rack['free_space']}/{rack['total_capacity']} free"
        else:
            stats_text = "No racks detected"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title("Analysis Summary")
        ax4.axis('off')
        
        plt.tight_layout()
        return fig
    
    def process_video_frame(self, video_path, frame_number=0, save_output=True):
        """Process specific frame from video"""
        print(f"🎬 Processing frame {frame_number} from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        # Jump to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ Error: Could not read frame {frame_number}")
            return None
        
        print(f"📸 Frame {frame_number} loaded, shape: {frame.shape}")
        
        # Run detection
        print("🔍 Running object detection...")
        detections = self.detect_objects(frame)
        print(f"Found {len(detections)} objects")
        
        # Simulate warehouse analysis
        print("🏭 Simulating warehouse rack analysis...")
        simulated_racks = self.simulate_warehouse_racks(frame, detections)
        print(f"Identified {len(simulated_racks)} potential racks")
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        annotated_frame = self.draw_detections(frame, detections, simulated_racks)
        
        if save_output:
            # Save annotated frame
            os.makedirs("output", exist_ok=True)
            output_path = f"output/visualization_frame_{frame_number:06d}.jpg"
            cv2.imwrite(output_path, annotated_frame)
            print(f"💾 Saved annotated frame: {output_path}")
            
            # Create matplotlib visualization
            fig = self.create_matplotlib_visualization(frame, detections, simulated_racks)
            plot_path = f"output/analysis_frame_{frame_number:06d}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"💾 Saved analysis plot: {plot_path}")
            
            # Save detection data
            analysis_data = {
                'frame_number': frame_number,
                'video_path': video_path,
                'timestamp': datetime.now().isoformat(),
                'raw_detections': detections,
                'warehouse_racks': simulated_racks,
                'summary': {
                    'total_racks': len(simulated_racks),
                    'total_capacity': sum(r['total_capacity'] for r in simulated_racks),
                    'total_free_space': sum(r['free_space'] for r in simulated_racks),
                    'average_utilization': sum(100 - r['empty_percentage'] for r in simulated_racks) / len(simulated_racks) if simulated_racks else 0
                }
            }
            
            json_path = f"output/analysis_data_frame_{frame_number:06d}.json"
            with open(json_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"💾 Saved analysis data: {json_path}")
        
        return {
            'frame': frame,
            'annotated_frame': annotated_frame,
            'detections': detections,
            'simulated_racks': simulated_racks
        }
    
    def process_multiple_frames(self, video_path, num_frames=5, interval=100):
        """Process multiple frames from video"""
        print(f"🎬 Processing {num_frames} frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        results = []
        for i in range(num_frames):
            frame_number = min(i * interval, total_frames - 1)
            result = self.process_video_frame(video_path, frame_number)
            if result:
                results.append(result)
        
        return results

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Warehouse Rack Visualization Tool")
    parser.add_argument("--video", default="video/VID_20250527_132714 (1).mp4", help="Path to video file")
    parser.add_argument("--frame", type=int, default=120, help="Frame number to analyze")
    parser.add_argument("--multiple", action="store_true", help="Process multiple frames")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    print("🎯 Warehouse Rack Visualization Tool")
    print("=" * 50)
    
    # Initialize visualization tool
    visualizer = WarehouseVisualizationTool(confidence_threshold=args.confidence)
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        # List available videos
        video_dir = "video"
        if os.path.exists(video_dir):
            videos = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if videos:
                print("📹 Available videos:")
                for i, video in enumerate(videos):
                    print(f"  {i+1}. {video}")
                args.video = os.path.join(video_dir, videos[0])
                print(f"🔄 Using first available video: {args.video}")
        
        if not os.path.exists(args.video):
            print("❌ No videos found")
            return
    
    # Process video
    if args.multiple:
        print("🔄 Processing multiple frames...")
        results = visualizer.process_multiple_frames(args.video, num_frames=3, interval=200)
        print(f"✅ Processed {len(results)} frames")
    else:
        print(f"🔄 Processing single frame {args.frame}...")
        result = visualizer.process_video_frame(args.video, args.frame)
        if result:
            print("✅ Frame processed successfully")
        else:
            print("❌ Failed to process frame")
    
    print("\n📁 Check the 'output' directory for visualization results")
    print("🖼️  View the generated images to see bounding box predictions")

if __name__ == "__main__":
    main()
