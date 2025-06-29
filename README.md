
# 🐦 Bird Identification System
<img src="https://github.com/user-attachments/assets/939fd404-9e5d-4bbe-9be9-469a59ab1fd4" alt="bis-logo" width="300"/>

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15.11-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange)

## Bird Identification System is a desktop application with a graphical interface for automatic identification of birds from photos, videos or real-time camera feeds. The system uses pre-trained computer vision models (YOLOv8 and CNN) to detect and classify birds, and provides the user with the ability to save results and provide feedback.

### 📌 Main features

* 🔍 Detection and classification of birds:
  - by **image**
  - by **video file**
  - in **real time** from a video stream
* 🗃️ Viewing and filtering the observation archive
* 📤 Provide feedback on identification results
* 🖥️ Simple and intuitive graphical interface in PyQt5

---

### 🚀 How to run

> ⚠️ For now, the application is launched without installation, via a Python script.

```bash
  git clone https://github.com/yourusername/bird-identification-system.git
  cd bird-identification-system
```

#### 1. Clone the repository:
```bash
  git clone https://github.com/yourusername/bird-identification-system.git
  cd bird-identification-system
```
    
#### 2. Create a virtual environment and activate it:
```bash
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
```
    
#### 3. Install dependencies:
```bash
  pip install -r requirements.txt
```

#### 4. Run the application:
```bash
  python script/main.py
```

---

### 🗂️ Project structure
```bash
bird-identification-system/
├── dataset/              # Data for training, testing, video examples, archive
├── gui/                  # Graphical interface (PyQt5)
├── model/                # Saved models (YOLOv8, CNN, class list)
├── script/               # Main run scripts (main.py, train.py, test.py)
├── src/                  # Basic logic: detection, classification, learning класифікація, навчання
├── requirements.txt      # Python dependencies
└── README.md             # Project description
```

---

### 🧠 Technical details
* YOLOv8 (yolov8n.pt) — for detecting birds in images and video
* CNN-model (bird_classification_model.h5) — for classifying cropped images
* Interaction with the user via PyQt5
* Archiving of results in the subdirectory dataset/observations/
* Optional user feedback with class refinement

---

### 🛠️ System requirements

> At the moment, the application works by running main.py. In the future, an .exe assembly for Windows will be implemented.

#### Recommended environment:
* Python 3.10
* Windows / Linux 
* Installed libraries from requirements.txt

#### Basic hardware:
* Processor: any Intel or AMD processor released in 2015 or later 
* RAM: 4 GB+.
* Video adapter: with OpenGL support
* Camera: USB camera (e.g. Logitech C270, Canon EOS 2000D etc.)

---

### 📦 Dependencies
#### Basic libraries (see requirements.txt):
* GUI: PyQt5, pygrabber
* CV/DL/ML: opencv-python, tensorflow, ultralytics, scikit-learn

---

### 📌 TODO / Future plans
* Creating an .exe assembly (Windows installer)
* Expansion of the dataset
* Improved classification accuracy using validated saved observations

---

#### Bachelor's project, National Aerospace University "Kharkiv Aviation Institute", 2025
