
import sys
sys.path.append('src')
import os
import yaml
from ultralytics import YOLO
from enhanced_model_trainer import EnhancedWarehouseModelTrainer

def train_empty_space_model():
    """Train a YOLOv8 model to detect empty spaces."""
    # Load the custom model configuration
    config_path = 'data/empty_space_data.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize the model trainer
    trainer = EnhancedWarehouseModelTrainer()
    trainer.training_config['data'] = config_path
    trainer.training_config['epochs'] = 50
    trainer.training_config['batch_size'] = 16
    trainer.training_config['project'] = 'empty_space_detector'

    # Start the training process
    model, results = trainer.train_enhanced_model()

    # Save the final model
    final_model_path = os.path.join('models', 'final', 'empty_space_model.pt')
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    results.save(final_model_path)

    print(f"Trained empty space model saved to {final_model_path}")

if __name__ == '__main__':
    train_empty_space_model()
