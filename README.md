
# Face Mask Detection in Video Streams using YOLOv8

## Project Overview

This project implements a computer vision–based system for detecting and classifying face mask usage in monocular video streams. The system was developed as part of a **Computer Vision course project at IU International University of Applied Sciences**. It aims to support automated monitoring of COVID-19 health protocols in indoor public spaces by identifying whether individuals are wearing face masks correctly, incorrectly, or not at all.

The solution leverages the **YOLOv8 object detection framework** to perform real-time face detection and mask classification on video data. The trained model processes video frames sequentially and produces annotated output videos showing bounding boxes and class labels.

---

## Objectives

* Detect human faces in video sequences using a deep learning–based object detector
* Classify mask usage into three categories:

  * `with_mask`
  * `without_mask`
  * `mask_worn_incorrectly`
* Evaluate model performance using precision, recall, and mAP metrics
* Demonstrate the system on real-world video data

---

## Dataset

A publicly available **bounding-box annotated face mask dataset** was used for training and validation. The dataset contains labeled facial regions corresponding to the three mask classes. Bounding-box annotations enable direct training of object detection models such as YOLOv8.

---

## Methodology

1. **Data Preparation**

   * Dataset split into training and validation sets
   * Annotations converted to YOLO format

2. **Model Training**

   * YOLOv8 model trained on the dataset using Ultralytics framework
   * Training performed on Google Colab with GPU acceleration

3. **Evaluation**

   * Model evaluated using precision, recall, mAP@50, and mAP@50–95

4. **Video Inference**

   * Monocular video sequences processed frame by frame
   * Bounding boxes and mask classification labels rendered on output videos

---

## Results

The trained model achieved strong detection performance across all classes, demonstrating reliable mask detection in video streams. The system performs effectively in real-world conditions, with most errors occurring under poor lighting or occlusion scenarios.

Annotated output videos are included to visually demonstrate the system’s performance.

---

## Repository Structure

```
├── data/                 # Dataset configuration files
├── runs/                 # Training and inference outputs
├── notebooks/            # Jupyter/Colab notebooks
├── videos/               # Input and output videos
├── README.md             # Project documentation
```

---

## Installation & Requirements

The project was developed using **Python 3.10+**.

Main dependencies:

* ultralytics
* opencv-python
* numpy
* matplotlib

Install dependencies:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

## Running the Project

### Training

```bash
yolo task=detect mode=train model=yolov8s.pt data=mask_dataset.yaml epochs=100 imgsz=640
```

### Video Inference

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.predict(source="input_video.avi", save=True)
```

---

## Demo Video

A demonstration video showing the system output is available via the link provided in the project report.

---

## Limitations & Future Work

* Performance may degrade under low-resolution or heavily occluded conditions
* Dataset size and class imbalance may affect detection accuracy

Future improvements include training on larger datasets, improving robustness under challenging conditions, and extending the system to monitor additional public safety protocols.

---

## Author

**Sunday Chibuike Onah**
Applied Artificial Intelligence
IU International University of Applied Sciences, Berlin

---

## License

This project is intended for **academic and educational purposes only**.


