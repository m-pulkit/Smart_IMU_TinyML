# Probabilistic TinyML for MEMS IMU Edge Inference

This repository contains an end-to-end Edge AI pipeline for real-time micro-gesture classification using uncalibrated 6-axis MEMS IMU data. It simulates a hardware-constrained environment (e.g., Smart IMU/System-in-Package) communicating over MQTT.

The project goes beyond standard time-series classification by incorporating Probabilistic Modeling and System Identification to calculate predictive epistemic uncertainty (Shannon Entropy) in real-time, rejecting false positives caused by physical sensor noise or out-of-distribution movements.

## The 3-Class Gesture Problem

Classifying human movement requires distinguishing deliberate kinetics from ambient mechanical noise. This project focuses on three specific states:

### Class 0: Idle / Null

**Behavior:** Background noise—typing, walking, or the phone resting at arbitrary orientations.

**Challenge:** The network must ignore constant gravity vectors and continuous low-frequency motion.

### Class 1: Upward Wrist Flick

**Behavior:** A sharp rotational lift, mimicking answering a phone call or waking a smartwatch.

**Challenge:** Initially, the neural network lazily overfits to the static gravity vector (e.g., detecting if the phone is just held vertically). Implementing a Dynamic DC-Blocker forces the model to learn the actual AC kinetic energy of the flick, making it 100% orientation-invariant.

### Class 2: Sideways Flat Tap

**Behavior:** The device is resting on a flat surface and is struck from the side, causing a sharp lateral impulse.

**Challenge:** A single tap versus a double tap or sliding scrape can look identical if the network's time dimension is crushed. We utilize Temporal Binned Global Average Pooling (GAP) to preserve the sequential impulse signature without the memory overhead of an LSTM.

## Core Features

### Hardware-Optimized 1D-CNN

Built using Depthwise Separable Convolutions and Temporal Binned GAP to maintain a footprint of < 8 KB Flash and < 15 KB SRAM.

### Dynamic DC-Blocking

Orientation-invariant feature engineering that removes the $9.81 \, m/s^2$ gravity vector and static gyroscope offsets dynamically, eliminating the need for memory-heavy global scalers.

### System Identification & Monte Carlo TTA

Dynamically identifies real-time sensor variance, injects Gaussian noise, and samples the predictive distribution to calculate epistemic uncertainty via Shannon Entropy.

## Live Inference Output & Hardware Profiling

During live emulation, the script buffers 1-second windows at 100Hz and executes Monte Carlo Test-Time Augmentation (TTA). The model outputs its confidence, the calculated Shannon Entropy (uncertainty), and the simulated hardware cost for the edge device.

Only events with an Entropy < 0.60 are accepted; highly uncertain predictions are rejected as physical noise.

```text
🚀 [EVENT TRIGGERED]: Gesture 2 (Confidence: 88.6%)
   |-> Predictive Entropy (Uncertainty): 0.4033
   |-> Simulated SRAM Footprint: ~12.8 KB
   |-> Computational Cost: ~38,400 MACs
--------------------------------------------------
🚀 [EVENT TRIGGERED]: Gesture 1 (Confidence: 87.9%)
   |-> Predictive Entropy (Uncertainty): 0.4038
   |-> Simulated SRAM Footprint: ~12.8 KB
   |-> Computational Cost: ~38,400 MACs
--------------------------------------------------
🚀 [EVENT TRIGGERED]: Gesture 1 (Confidence: 74.1%)
   |-> Predictive Entropy (Uncertainty): 0.5730
   |-> Simulated SRAM Footprint: ~12.8 KB
   |-> Computational Cost: ~38,400 MACs
```

## Quick Start Guide

### 1. Data Ingestion

Start a local Mosquitto broker. Use an MQTT sensor app (e.g., Sensor Spot) to stream uncalibrated IMU data over MQTT at 100Hz.

```bash
python mqtt_logger.py
```

Press `1` or `2` on your keyboard while performing gestures to label the incoming stream for supervised learning.

### 2. Model Training & Export

Open the `Notebooks/` directory or run the Python scripts directly.

Run `TinyML_Gesture_CNN.ipynb` to train the depthwise separable 1D-CNN using dynamic DC-blocking.

Run `Export_to_Edge.ipynb` to export the PyTorch graph to ONNX (`tiny_gesture.onnx`) and generate the `representative_dataset.bin` required for downstream INT8 Post-Training Quantization (PTQ).

### 3. Live Edge Emulation

Run the probabilistic inference wrapper to classify live movements over MQTT:

```bash
python live_edge_emulator.py
```