# Warehouse Computer Vision Project

This project provides a complete system for warehouse free space measurement and management using computer vision and a modern web interface.

## Features
- **YOLOv8-based object detection** for racks, pallet slots, and empty spaces
- **Automated frame extraction** from warehouse surveillance videos
- **Synthetic annotation generation** for training data
- **Interactive web frontend** for data entry, analysis, and reporting
- **Backend API** with data storage, Excel export, and REST endpoints
- **Easy-to-use training and evaluation scripts**

## Directory Structure
- `src/` - Core Python scripts for training, calibration, and visualization
- `backend/` - Flask backend API and database
- `frontend/` - Web UI (HTML/JS/CSS)
- `data/` - Training images, labels, and configuration
- `models/` - (Ignored) Model weights and checkpoints
- `video/` - (Ignored) Raw video files
- `output/`, `exports/` - (Ignored) Generated outputs and reports

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/sahit1011/Warehouse_CV_clean.git
   cd Warehouse_CV_clean
   ```
2. **Install Python dependencies:**
   ```sh
   pip install -r backend/requirements.txt
   ```
3. **Run the backend server:**
   ```sh
   python backend/app.py
   ```
4. **Access the frontend:**
   Open [http://localhost:5001/](http://localhost:5001/) in your browser.

## Usage
- Use the web UI to add, view, and analyze warehouse storage data.
- Train and evaluate models using scripts in `src/`.
- Export data and reports from the frontend.

## Notes
- Large files (videos, models, outputs) are **not tracked** in this repository. Add your own data as needed.
- For training your own models, place videos in the `video/` folder and follow the instructions in `src/train_model.py`.

## License
MIT
