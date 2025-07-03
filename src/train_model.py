#!/usr/bin/env python3
"""
train_model.py
Enhanced script to train a YOLOv8 model for warehouse free space detection.
This script includes data preparation, model training, and evaluation capabilities.
"""

import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil

class WarehouseModelTrainer:
    def __init__(self, project_root="."):
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, "data")
        self.models_dir = os.path.join(project_root, "models")
        self.video_dir = os.path.join(project_root, "video")

        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "labels", "val"), exist_ok=True)

    def extract_frames_from_videos(self, frame_interval=60):
        """Extract frames from warehouse videos for training data"""
        print("📹 Extracting frames from warehouse videos...")

        if not os.path.exists(self.video_dir):
            print(f"❌ Video directory '{self.video_dir}' not found")
            return False

        video_files = [f for f in os.listdir(self.video_dir)
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print(f"❌ No video files found in '{self.video_dir}'")
            return False

        frame_count = 0
        for video_file in video_files:
            video_path = os.path.join(self.video_dir, video_file)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ Could not open {video_file}")
                continue

            print(f"📸 Processing {video_file}")
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval == 0:
                    # Save frame for annotation
                    frame_filename = f"frame_{video_file}_{frame_num:06d}.jpg"
                    frame_path = os.path.join(self.data_dir, "images", "train", frame_filename)
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1

                frame_num += 1

            cap.release()

        print(f"✅ Extracted {frame_count} frames for training")
        return True

    def create_dataset_config(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': os.path.abspath(self.data_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,  # Number of classes
            'names': ['empty_slot', 'occupied_slot', 'rack']
        }

        config_path = os.path.join(self.data_dir, "data.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✅ Created dataset config: {config_path}")
        return config_path

    def list_images(self, image_dir):
        """List available images in directory"""
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)

        images = [f for f in os.listdir(image_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(images)} images in {image_dir}")
        return images

    def train_yolo_model(self, epochs=50, model_size='yolov8n.pt', img_size=640):
        """Train YOLO model with warehouse data"""
        print("🚀 Starting YOLOv8 training...")

        # Create dataset config
        data_yaml = self.create_dataset_config()

        # Check if we have training images
        train_images = self.list_images(os.path.join(self.data_dir, "images", "train"))
        if len(train_images) == 0:
            print("⚠️  No training images found. Extracting frames from videos...")
            self.extract_frames_from_videos()
            train_images = self.list_images(os.path.join(self.data_dir, "images", "train"))

        if len(train_images) == 0:
            print("❌ No training data available. Please add videos or images.")
            return None

        # Initialize model
        model = YOLO(model_size)

        # Train the model
        try:
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                patience=10,
                save=True,
                project=self.models_dir,
                name='warehouse_detection'
            )
            print("✅ Training complete!")
            return model
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None

    def evaluate_model(self, model, test_image_path=None):
        """Evaluate trained model"""
        if model is None:
            print("❌ No model provided for evaluation")
            return None

        if test_image_path is None:
            # Use a random training image for evaluation
            train_dir = os.path.join(self.data_dir, "images", "train")
            images = self.list_images(train_dir)
            if images:
                test_image_path = os.path.join(train_dir, images[0])
            else:
                print("❌ No images available for evaluation")
                return None

        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return None

        print(f"🔍 Evaluating model on {test_image_path}")
        try:
            results = model(test_image_path)

            # Save results
            output_path = os.path.join(self.models_dir, "evaluation_result.jpg")
            results[0].save(output_path)
            print(f"💾 Evaluation result saved: {output_path}")

            return results
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None

    def save_model(self, model, model_name="warehouse_yolov8.pt"):
        """Save trained model"""
        if model is None:
            print("❌ No model to save")
            return False

        save_path = os.path.join(self.models_dir, model_name)
        try:
            model.save(save_path)
            print(f"✅ Model saved: {save_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

    def create_synthetic_annotations(self):
        """Create synthetic annotations based on existing detection pipeline"""
        print("🤖 Creating synthetic annotations using existing detection pipeline...")
        
        from main import WarehouseAnalyzer
        
        # Initialize analyzer
        analyzer = WarehouseAnalyzer()
        if not analyzer.load_model():
            print("❌ Failed to load detection model for annotation")
            return False
        
        train_dir = os.path.join(self.data_dir, "images", "train")
        labels_dir = os.path.join(self.data_dir, "labels", "train")
        
        images = self.list_images(train_dir)
        annotation_count = 0
        
        for image_file in images:
            image_path = os.path.join(train_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            # Run detection on the image
            try:
                results = analyzer.model(image)
                
                # Process detections and create YOLO format annotations
                detections = analyzer.process_detections(results[0])
                
                if detections:
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_file)
                    
                    with open(label_path, 'w') as f:
                        for det in detections:
                            # Convert to YOLO format (normalized coordinates)
                            h, w = image.shape[:2]
                            x1, y1, x2, y2 = det.bbox
                            
                            # Normalize coordinates
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            # Map class name to ID
                            class_id = self.get_class_id(det.class_name)
                            
                            # Write YOLO format: class_id x_center y_center width height
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    annotation_count += 1
                    
            except Exception as e:
                print(f"⚠️  Failed to process {image_file}: {e}")
                continue
        
        print(f"✅ Created {annotation_count} annotation files")
        return annotation_count > 0
    
    def get_class_id(self, class_name):
        """Map class name to ID for warehouse objects"""
        class_map = {
            'rack': 0,
            'pallet': 1,
            'shelf': 2,
            'empty_space': 3,
            'obstruction': 4,
            'person': 5,
            'forklift': 6
        }
        return class_map.get(class_name, 0)
    
    def prepare_training_data(self):
        """Complete training data preparation pipeline"""
        print("🔧 Preparing complete training dataset...")
        
        # 1. Extract frames if needed
        train_images = self.list_images(os.path.join(self.data_dir, "images", "train"))
        if len(train_images) == 0:
            print("📹 Extracting frames from videos...")
            if not self.extract_frames_from_videos():
                return False
        
        # 2. Create synthetic annotations
        if not self.create_synthetic_annotations():
            return False
        
        # 3. Split data into train/val
        self.split_dataset()
        
        return True
    
    def split_dataset(self, val_split=0.2):
        """Split dataset into train and validation sets"""
        print(f"📊 Splitting dataset (train: {1-val_split:.0%}, val: {val_split:.0%})...")
        
        train_img_dir = os.path.join(self.data_dir, "images", "train")
        train_lbl_dir = os.path.join(self.data_dir, "labels", "train")
        val_img_dir = os.path.join(self.data_dir, "images", "val")
        val_lbl_dir = os.path.join(self.data_dir, "labels", "val")
        
        # Get all annotated images
        images = []
        for img_file in os.listdir(train_img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if os.path.exists(os.path.join(train_lbl_dir, label_file)):
                    images.append(img_file)
        
        # Shuffle and split
        np.random.shuffle(images)
        val_count = int(len(images) * val_split)
        val_images = images[:val_count]
        
        # Move validation images and labels
        for img_file in val_images:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            
            # Move image
            shutil.move(
                os.path.join(train_img_dir, img_file),
                os.path.join(val_img_dir, img_file)
            )
            
            # Move label
            shutil.move(
                os.path.join(train_lbl_dir, label_file),
                os.path.join(val_lbl_dir, label_file)
            )
        
        print(f"✅ Moved {val_count} images to validation set")
        print(f"📊 Final dataset: {len(images) - val_count} train, {val_count} val")

def main():
    """Main function to train warehouse detection model"""
    print("🏭 Enhanced Warehouse YOLO Model Training System")
    print("=" * 60)

    # Initialize trainer
    trainer = WarehouseModelTrainer()

    # 1. Prepare complete training data
    print("\n📹 Step 1: Preparing training data...")
    if not trainer.prepare_training_data():
        print("❌ Failed to prepare training data")
        return
    trainer.extract_frames_from_videos(frame_interval=90)  # Every 3 seconds at 30fps

    # 2. List available images
    train_dir = os.path.join(trainer.data_dir, 'images', 'train')
    images = trainer.list_images(train_dir)

    if len(images) == 0:
        print("❌ No training images available. Please add videos to the 'video' directory.")
        return

    print(f"\n📊 Found {len(images)} training images")
    print("⚠️  Note: For actual training, you'll need to annotate these images with bounding boxes")
    print("   Use tools like LabelImg or Roboflow to create YOLO format annotations")

    # 3. Train model (only if annotations exist)
    labels_dir = os.path.join(trainer.data_dir, 'labels', 'train')
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] if os.path.exists(labels_dir) else []

    if len(label_files) > 0:
        print(f"\n🚀 Step 2: Training model with {len(label_files)} annotated images...")
        model = trainer.train_yolo_model(epochs=10)  # Reduced epochs for demo

        if model:
            # 4. Evaluate model
            print("\n🔍 Step 3: Evaluating model...")
            trainer.evaluate_model(model)

            # 5. Save model
            print("\n💾 Step 4: Saving model...")
            trainer.save_model(model)

            print("\n✅ Training pipeline complete!")
        else:
            print("❌ Training failed")
    else:
        print("\n⚠️  No annotation files found. Skipping training.")
        print("   To train a model:")
        print("   1. Annotate the extracted frames using LabelImg or similar tool")
        print("   2. Save annotations in YOLO format to data/labels/train/")
        print("   3. Run this script again")

    print(f"\n📁 Training data location: {trainer.data_dir}")
    print(f"📁 Models will be saved to: {trainer.models_dir}")


if __name__ == "__main__":
    main()