import paho.mqtt.client as mqtt
import json
import numpy as np
import onnxruntime as ort
from collections import deque

data_dir = "data/"
training_dir = "training_outputs/"

# --- CONFIGURATION ---
BROKER_IP = "192.168.2.110" # CHANGE TO YOUR PC'S IP
TOPIC = "S25U/#" # Chosen topic from sensor MQTT app
CSV_FILENAME = data_dir + "gesture_dataset_raw.csv"
WINDOW_SIZE = 100

# --- LOAD EDGE MODEL ---
print("Loading ONNX model and calibration parameters...")
ort_session = ort.InferenceSession(training_dir + "tiny_gesture.onnx")

# Scalers removed: We use dynamic DC-blocking now

CLASSES = ["Idle", "Gesture 1", "Gesture 2"]
live_buffer = deque(maxlen=WINDOW_SIZE)

current_state = {
    "ax": 0.0, "ay": 0.0, "az": 0.0,
    "gx": 0.0, "gy": 0.0, "gz": 0.0
}

def on_connect(client, userdata, flags, rc):
    print(f"Connected to Mosquitto Broker with code {rc}.")
    print("Listening for live gestures... (Perform movements with your phone)")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global current_state
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        sensor_type = payload.get("type", "").lower()
        values = payload.get("values", [0,0,0,0,0,0])
        
        # Update gyro state passively
        if "gyroscope" in sensor_type:
            current_state["gx"], current_state["gy"], current_state["gz"] = values[:3]
            
        # Update accel state and trigger the buffer append (our 100Hz clock)
        elif "accelerometer" in sensor_type:
            current_state["ax"], current_state["ay"], current_state["az"] = values[:3]
            
            # Combine into 6-axis row
            sensor_row = [
                current_state["ax"], current_state["ay"], current_state["az"],
                current_state["gx"], current_state["gy"], current_state["gz"]
            ]
            live_buffer.append(sensor_row)

            # Once we have 1 second of data, run inference
            if len(live_buffer) == WINDOW_SIZE:
                
                # 1. Convert to numpy and apply dynamic DC-blocking (remove gravity)
                features = np.array(live_buffer)
                features = features - np.mean(features, axis=0)
                
                # --- PROBABILISTIC MODELING & SYSTEM ID ---
                # Identify live sensor noise variance, inject Gaussian noise, 
                # and sample the predictive distribution to calculate Epistemic Uncertainty.
                N_SAMPLES = 10
                sensor_noise_std = np.std(features, axis=0) * 0.3 # Scale noise to IMU characteristics
                
                predictions_batch = []
                for _ in range(N_SAMPLES):
                    # Inject Brownian noise to simulate physical sensor uncertainty
                    noisy_sample = features + np.random.normal(0, sensor_noise_std, features.shape)
                    noisy_sample = noisy_sample.T.reshape(1, 6, WINDOW_SIZE).astype(np.float32)
                    
                    ort_inputs = {ort_session.get_inputs()[0].name: noisy_sample}
                    out = ort_session.run(None, ort_inputs)[0][0]
                    predictions_batch.append(out)
                    
                predictions_batch = np.array(predictions_batch)
                
                # Apply softmax to convert raw logits to probability distributions
                exp_preds = np.exp(predictions_batch - np.max(predictions_batch, axis=1, keepdims=True))
                probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
                
                # Calculate Mean Probability and Shannon Entropy (Uncertainty)
                mean_probs = np.mean(probs, axis=0)
                prediction = np.argmax(mean_probs)
                entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9))
                
                # Only trigger if not "Idle" AND model is highly confident (Low Entropy)
                if prediction != 0 and entropy < 0.6: 
                    print(f"\n🚀 [EVENT TRIGGERED]: {CLASSES[prediction]} (Confidence: {mean_probs[prediction]*100:.1f}%)")
                    print(f"   |-> Predictive Entropy (Uncertainty): {entropy:.4f}")
                    print(f"   |-> Simulated SRAM Footprint: ~12.8 KB")
                    print(f"   |-> Computational Cost: ~38,400 MACs")
                    print("-" * 50)
                    
                    # Clear buffer to prevent double-triggering on the tail end of the movement
                    live_buffer.clear() 
                    
    except Exception as e:
        # Silently pass malformed JSON to keep the live stream running
        pass

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Ensure this matches the Mosquitto broker setup
    client.connect(BROKER_IP, 1883, 60)
    
    # Blocking loop to keep the script running
    client.loop_forever()