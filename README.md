# On-Device PPG Sensor-Contact Detection with MAX30102 and ESP32 (TinyML)

An end-to-end TinyML pipeline that takes raw photoplethysmography (PPG) readings from a MAX30102 optical sensor, trains a tiny neural network, quantizes it to int8, and compiles it into C for inference on an ESP32.

The current model detects **whether a finger is in contact with the sensor**. Most wearable health pipelines need this first step before they compute heart rate or SpO₂, because readings taken without skin contact produce misleading vitals. The project is also a working template for the full sensor → model → microcontroller workflow, which I am extending to heart-rate estimation (see [Roadmap](#roadmap)).

![Circuit diagram](https://github.com/user-attachments/assets/98f4f4c1-8541-48d1-b8bb-8164c2c55a85)

## Results at a glance

| Item | Value |
|---|---|
| Sensor | MAX30102 (IR + Red channels) over I²C |
| Dataset | 2,049 self-collected samples (1,639 train / 410 test) |
| Model | Dense(16, ReLU) → Dense(1, sigmoid), **65 parameters** |
| Quantization | Full integer (int8) via TFLite post-training quantization |
| Model size on device | **2,496 bytes** (`model_simple.h`) |
| Test accuracy | 100% on held-out samples (see note below) |
| Inference latency on ESP32 | _TODO: measure with `micros()`_ |

**A note on the 100% accuracy.** The labels come from k-means clustering (k = 2) on the raw IR/Red values, and the network learns to reproduce that split. The two clusters line up with "finger on sensor" (IR ≈ 65k–118k counts) and "no contact" (IR in the hundreds), which are easy to separate. The perfect score therefore reflects an easy task, not a hard one. The value of this project is the deployment pipeline, which carries over directly to harder physiological tasks.

## How it works

1. **Data acquisition.** `data_logger_ESP32.ino` initializes the MAX30102 on the ESP32's I²C pins (SDA 21, SCL 22) and streams `IR,Red` pairs over serial. `Collecting_Sensor_data.py` logs them to CSV on the host.
2. **Labeling.** k-means (k = 2) groups the samples into contact and no-contact clusters, and these become the training targets.
3. **Preprocessing.** Min-max normalization of both channels (IR: 0–117,792; Red: 355–81,903). The device must apply the same scaling before inference.
4. **Training.** A 65-parameter dense network trained with Adam (lr = 0.001) and binary cross-entropy for 50 epochs, using an 80/20 train/test split plus a 20% validation split.
5. **Quantization.** The TFLite converter with a representative dataset generator performs full int8 post-training quantization.
6. **Deployment.** The `.tflite` model is converted to a C byte array (`model_simple.h`) and compiled into the ESP32 firmware with TensorFlow Lite for Microcontrollers.

## Repository contents

| File | Purpose |
|---|---|
| `data_logger_ESP32.ino` | ESP32 firmware that streams raw IR/Red values |
| `Collecting_Sensor_data.py` | Host-side serial → CSV logger |
| `max30102_model.ipynb` | Labeling, normalization, training, quantization, C-array export |
| `model_simple.h` | Quantized model as a C array (2,496 bytes) |
| `Collected MAX30102 Sensor Data Visualization.png` | Raw signal plot |
| `Loss And Accuracy Curve.png` | Training curves |

## Running it

**Hardware:** ESP32 dev board, MAX30102 module, jumper wires, USB cable.
**Software:** Arduino IDE with the ESP32 board package and the SparkFun MAX3010x library; Python 3.10 with NumPy, pandas, scikit-learn, Matplotlib, TensorFlow.

1. Flash `data_logger_ESP32.ino`, set your serial port in `Collecting_Sensor_data.py`, and log data with and without a finger on the sensor.
2. Run `max30102_model.ipynb` to train, quantize, and export `model_simple.h`.
3. Include `model_simple.h` in your TFLite Micro sketch, normalize each reading with the min/max values above, and run inference.

## Limitations

- Samples are logged at 1 Hz. A pulse waveform needs roughly 25–100 Hz, so this data cannot resolve individual heartbeats.
- The model classifies single samples and does not use time windows, so it has no access to waveform shape.
- Labels come from unsupervised clustering, not ground truth.
- All data comes from one person and one session.

## Roadmap

- [ ] Log at 100 Hz and segment the signal into 4–8 s windows
- [ ] Band-pass filter (~0.5–4 Hz) and peak-based heart-rate estimation as a baseline
- [ ] Record reference heart rate from a commercial pulse oximeter or smartwatch
- [ ] Train a small 1D CNN for heart-rate estimation or signal-quality classification, and report MAE in BPM
- [ ] Add the on-device inference sketch, with measured latency and RAM use
- [ ] Extend to SpO₂ estimation using the IR/Red ratio

## Author

Jeffery Owusu Cobbold · B.Sc. Engineering Physics, University of Cape Coast
