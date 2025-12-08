# On-Device Heart-Rate Analysis Using MAX30102 and ESP32 (TinyML Project)

## Overview
This project implements a full TinyML workflow for physiological signal analysis. Data from the MAX30102 heart-rate sensor is collected using an ESP32, preprocessed, and used to train a lightweight TensorFlow model. The trained model is then converted to TensorFlow Lite, quantized to int8, and deployed on the ESP32 for efficient real-time inference.

---

## Circuit Diagram

<img width="2019" height="948" alt="HR_SPO2_circuit_diagram_bb" src="https://github.com/user-attachments/assets/98f4f4c1-8541-48d1-b8bb-8164c2c55a85" />


---

## Features
- Real-time IR and Red signal acquisition  
- Dataset preprocessing and cleaning  
- Lightweight TensorFlow model training  
- Full integer quantization (int8)  
- On-device inference running on ESP32  
- Offline health monitoring without cloud dependency  

---

## Hardware Requirements
- ESP32 Development Board  
- MAX30102 Heart-Rate & SpO₂ Sensor  
- Jumper wires  
- USB cable  

---

## Software Requirements
- Arduino IDE  
- ESP32 board package  
- MAX3010x library  
- Python (NumPy, Pandas, Matplotlib)  
- TensorFlow / TensorFlow Lite  


---

## System Workflow

### 1. Data Collection
The ESP32 streams IR and Red values from MAX30102 via I²C and logs samples for training.

### 2. Preprocessing
- Noise reduction  
- Normalization  
- Segmentation into fixed windows  
- Labeling (if classification)

### 3. Model Training
A compact TensorFlow model is trained on windowed sensor data. Emphasis is placed on memory efficiency and fast inference.

### 4. TFLite Conversion & Quantization
- Convert SavedModel → `.tflite`  
- Apply full integer quantization (int8)  
- Validate with TFLite Interpreter  

### 5. Deployment
The quantized model is compiled into the Arduino sketch using the TFLite Micro library and executed on the ESP32.






