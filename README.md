# Ship Detection and Distance Estimation Using Stereo Vision and YOLO

## 1. Introduction
This project presents a real-time system for detecting ships and estimating their distance using stereo vision and deep learning. The system integrates stereo camera calibration, depth estimation, object detection, and multi-object tracking.

The objective of this research is to support maritime monitoring by providing accurate distance estimation and collision risk assessment.

---

## 2. System Features
- Stereo camera calibration using fisheye model
- Real-time disparity and depth estimation
- Ship detection using YOLO (Ultralytics)
- Multi-object tracking using ByteTrack
- Distance smoothing based on object ID
- Time-to-Collision (TTC) estimation
- Visual warning system for collision risk

---

## 3. Methodology

### 3.1 Stereo Vision
Stereo images are captured from left and right cameras. The images are rectified using calibration parameters, and disparity is computed using the StereoSGBM algorithm. Depth information is obtained through reprojection.

### 3.2 Object Detection
A YOLO-based model is used to detect ships in real-time. Each detected object is represented by a bounding box.

### 3.3 Multi-Object Tracking
ByteTrack is used to assign unique IDs to detected objects, enabling consistent tracking across frames.

### 3.4 Distance Estimation
Depth values are extracted from the disparity map. Median filtering is applied within the bounding box region to reduce noise, followed by temporal smoothing per tracked object.

### 3.5 Time-to-Collision (TTC)
Velocity is estimated from changes in depth over time. TTC is computed as:

TTC = Distance / Velocity

### 3.6 Warning System
The system classifies collision risk based on TTC:
- TTC < 3 seconds: High risk
- 3 ≤ TTC < 7 seconds: Medium risk
- TTC ≥ 7 seconds: Low risk

---

## 4. Installation

### 4.1 Clone Repository
```bash
git clone https://github.com/your-username/ship-detection-stereo-vision.git
cd ship-detection-stereo-vision
