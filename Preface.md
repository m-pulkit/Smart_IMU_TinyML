# Engineering Background & Architectural Decisions

This document outlines the technical rationale behind the model architecture and data processing pipelines, specifically tailored for deployment on resource-constrained micro-DSPs (e.g., embedded sensor hubs).

## 1. Defeating Hardware Constraints (SRAM & Flash)

Standard Time-Series classification often relies on LSTMs or deep CNNs followed by massive dense layers. In a TinyML context (target: 32KB SRAM, 128KB Flash, ~50$\mu$W power budget), these architectures fail.

### Architectural Solutions

#### Depthwise Separable Convolutions

Splitting standard convolutions into a spatial filter and a channel mixer reduces Multiply-Accumulate (MAC) operations by nearly 80%.

#### Temporal Binned GAP

Replacing the Flatten and Linear layers with an `AdaptiveAvgPool1d(4)` layer collapses the time dimension into 4 distinct sequential bins. This preserves the temporal sequence of the data (e.g., distinguishing a single tap from a double tap) while stripping out over 100,000 parameters compared to a standard Dense layer.

## 2. Sensor Physics: Dynamic DC-Blocking vs. Static Scalers

Initial iterations of this model utilized global StandardScaler transformations ($z = \frac{x - \mu}{\sigma}$). This introduced two critical flaws in an embedded context:

### Memory Cost

Storing means and variances for scaling consumes valuable MCU memory.

### Orientation Bias

The uncalibrated accelerometer fundamentally measures the gravity vector. If trained with static scaling, the neural network lazily learns the static orientation of the device (e.g., Y-axis dominant) rather than the AC kinetic energy of the gesture.

### Solution

We implemented a per-window Dynamic DC-Blocker ($x' = x - \mu_{window}$). This dynamically subtracts the rolling mean of the 1-second window. It effectively high-pass filters the data, stripping out the static gravity vector and any slow-moving gyroscope temperature drift, making the model 100% orientation-invariant.

## 3. Probabilistic Modeling & System Identification

MEMS sensors are inherently noisy, subject to thermomechanical drift, vibration rectification, and angle random walk. Deterministic neural networks are brittle to this noise, often confidently predicting a class on pure static.

To make the edge deployment robust, we implemented Monte Carlo Test-Time Augmentation (TTA):

### System Identification

Upon receiving a 1-second window, the emulator calculates the instantaneous variance of the raw sensor data to identify the current physical noise floor.

### Stochastic Forward Passes

We inject Gaussian noise—scaled precisely to the identified sensor variance—into $N=10$ copies of the window.

### Shannon Entropy

By passing all $N$ noisy copies through the ONNX runtime, we generate a probability distribution of the predictions. We calculate the Shannon Entropy ($H = -\sum p \log p$). If the entropy exceeds our confidence threshold, the model rejects the input as hardware noise or an out-of-distribution movement.

This provides a lightweight, mathematically sound alternative to computationally heavy Bayesian Neural Networks, bringing true confidence intervals to the absolute edge.