# 🚧 Pothole-Detection-and-Anaysis-With-Machine-Learning-and-Computer-Vision

## Overview

This project is an advanced computer vision system that detects, tracks, and analyzes potholes in real-time from video footage. Using YOLOv8 segmentation and custom tracking algorithms, it provides:

- Real-time pothole visualization
- Size and risk classification
- Comprehensive analytics dashboard
- Automated report generation

## Key Features

- 🎯 **Accurate Detection**: YOLOv8 segmentation for precise pothole identification
- 📏 **Size Classification**: Small, Medium, Large based on area thresholds
- ⚠️ **Risk Assessment**: Low, Medium, High based on size and position
- 🔄 **Object Tracking**: Consistent ID assignment across frames
- 📊 **Dashboard**: Real-time statistics display
- 📝 **Automated Reporting**: Detailed analysis with severity ratings

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics
- NumPy

## Installation

```bash
# Clone repository
git clone https://github.com/FaisalAhmed21/Pothole-Detection-System-With-Machine-Learning-and-Computer-Vision.git
cd Pothole-Detection-System-With-Machine-Learning-and-Computer-Vision

# Install dependencies
pip install -r requirements.txt

# Download pretrained model
wget https://github.com/FaisalAhmed21/Pothole-Detection-System-With-Machine-Learning-and-Computer-Vision/releases/download/v1.0/best_02.pt
```

## Usage

```bash
# Run the pothole detection system
python test.py
```

## Controls

* Press **'q'** to exit and generate report

## Visual Indicators

| Element | Description | Visual |
|---------|-------------|--------|
| **Low-risk potholes** | Small potholes away from road center | 🟢 Green overlay |
| **Medium-risk potholes** | Medium potholes or small ones near center | 🟠 Orange overlay |
| **High-risk potholes** | Large potholes or critical position | 🔴 Red overlay |

## Outputs

### Real-time Visualization

* Color-coded pothole overlays (green/orange/red)
* Tracking IDs and risk levels displayed
* Safe path indicators (yellow lines)
* Statistics dashboard (top-left corner):
  * Current frame pothole counts
  * Total unique potholes detected
  * Size distribution (Small/Medium/Large)

### Generated Analysis Report

The system generates `pothole_analysis_report_<timestamp>.txt` containing:

1. **Video metadata** (filename, duration, frames analyzed)
2. **Pothole statistics:**
   * Size distribution
   * Risk classification
   * Average pothole area
3. **Road condition severity rating** (0-10 scale)
4. **Maintenance recommendations**

## Configuration

You can modify the following parameters in the script:

- `small_threshold = 5000`: Area threshold for small potholes
- `large_threshold = 15000`: Area threshold for large potholes
- `proximity_threshold = 400`: Distance for marking potholes
- `coordinate_threshold = 50`: Distance for tracking same pothole

