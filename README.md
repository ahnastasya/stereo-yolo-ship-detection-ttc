# Real-Time Ship Detection and Distance Estimation using Stereo Vision and YOLO

## Overview
This project implements a real-time computer vision system for detecting and tracking ships while estimating their distance using a stereo camera setup.

The system integrates:
- YOLOv12 for object detection
- ByteTrack for multi-object tracking
- Stereo fisheye camera calibration
- Depth estimation using StereoSGBM
- Time-to-Collision (TTC) analysis for safety monitoring

This project is developed as part of an undergraduate thesis in maritime engineering.

---

## Features
- Real-time ship detection
- Multi-object tracking with persistent IDs
- Depth estimation using stereo vision
- Distance smoothing per tracked object
- Time-to-Collision (TTC) calculation
- Visual warning system (Safe / Warning / Danger)

---

## System Pipeline
1. Capture stereo image (left-right camera)
2. Rectification using fisheye calibration
3. Disparity computation (StereoSGBM)
4. Depth reconstruction (3D projection)
5. Object detection (YOLO)
6. Object tracking (ByteTrack)
7. Distance estimation per object
8. TTC calculation
9. Visualization and warning system

---

## Project Structure
