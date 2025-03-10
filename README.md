# Traffic Sign Recognition using YOLO and CNN

This project implements a **real-time traffic sign recognition system** using **YOLOv4-tiny** for sign detection and a **CNN model** for classification. The system processes video input to detect and recognize traffic signs, enabling applications such as autonomous driving and driver assistance systems.

## Features
- **Traffic sign detection** using YOLOv4-tiny.
- **Sign classification** using a CNN trained on the German Traffic Sign dataset.
- **Real-time processing** from video files or live webcam feed.
- **Sign recognition visualization** with an overlay on the detected signs.

## Installation & Setup
To run the project, ensure you have the required dependencies installed.

### Prerequisites
- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy

### Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Ensure the necessary models and configuration files are in place:
   - **YOLOv4-tiny weights and config** (`v1.weights`, `detector_yolov4_tiny.cfg`)
   - **CNN model for classification** (`German_Winning_99_75.h5`)
   - **Class label file** (`classes.names`)

## Usage

### Running Traffic Sign Recognition on a Video
```sh
python Final_output.py
```
This script processes an input video, detects traffic signs, and classifies them.

### Running Traffic Sign Recognition with Live Webcam Feed
```sh
python webcam_stream.py
```
This script performs the same detection and classification but with real-time webcam input.

## Project Structure
```
│── Final_output.py         # Main pipeline script for traffic sign detection & recognition
│── webcam_stream.py        # Runs the same pipeline with webcam input
│── video_test.py           # Testing script for processing video input
│── recognition_test.py     # Tests classification model separately
│── recog_test_v2.py        # Additional recognition testing script
│── ROI_test.py             # Tests region-of-interest extraction
│── models/
│   ├── German_Winning_99_75.h5   # CNN model for classification
│── yolo/
│   ├── v1.weights         # YOLOv4-tiny weights
│   ├── detector_yolov4_tiny.cfg  # YOLO config file
│   ├── classes.names      # Label names for YOLO detection
│── images/
│   ├── no_sign.jpg        # Default image when no sign is detected
│── videos/
│   ├── input_video_7.mp4  # Sample input video
│── README.md              # Project documentation
```

## References & Citation
This implementation is based on:

> Abdel-Salam, R., Mostafa, R., & Abdel-Gawad, A.H. (2022).  
> "RIECNN: real-time image enhanced CNN for traffic sign recognition."  
> *Neural Computing & Applications, 34*, 6085–6096.  
> DOI: [10.1007/s00521-021-06762-5](https://doi.org/10.1007/s00521-021-06762-5)

If you use this project, please consider citing the above paper.

## Credits
- **YOLOv4-tiny model** for object detection.
- **CNN model** trained on the **German Traffic Sign Dataset**.
- Implementation and improvements by [Hadiya Sajjad](https://github.com/Hadiya-Sajjad).

---
🚀 *Happy Coding! Feel free to contribute and improve this project!*  
