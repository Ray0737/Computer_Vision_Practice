# OpenCV Tutorial

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for Computer Vision tasks ranging from basic image processing to real-time motion analysis and facial identification.

---

## 🛠 Features

### 🖼️ Image & Video Editing
* **Pre-processing:** Grayscaling, Gaussian Blur, Canny Edge Detection, and Dilations.
* **Transformations:** Resizing, cropping, rotating, and Perspective Wrapping.
* **Bitwise Operations:** Masking and combining images for overlay effects.

### 👤 Face Detection & Recognition
* **Haar Cascades:** Real-time detection of faces, eyes, and smiles.
* **LBPH (Local Binary Patterns Histograms):** Train the system to recognize specific individuals by name.
* **Deep Learning (DNN):** Implementation of SSD-based face detection for better accuracy in low light.

### 🏃 Motion Detection & Security
* **Frame Differencing:** Identify movement by calculating the absolute difference between frames.
* **Contour Detection:** Automatic bounding-box creation around moving entities.
* **Logging:** Logic to trigger "Record" or "Alert" when motion exceeds a specific threshold.

---

## 📂 Repository Structure

```text
.
├── 📁 01-Image-Basics        # Resizing, cropping, and color spaces
├── 📁 02-Video-Processing    # FPS handling, writing video files, overlays
├── 📁 03-Face-Recognition    # Training scripts and detection logic
├── 📁 04-Motion-Detection    # Background subtraction and security logic
├── 📁 assets                 # Sample images/videos for testing
├── 📁 models                 # Pre-trained .xml and .caffemodel files
├── requirements.txt          # Project dependencies
└── main.py                   # Unified entry point (optional)
