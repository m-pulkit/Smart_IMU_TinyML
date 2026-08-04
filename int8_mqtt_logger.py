import paho.mqtt.client as mqtt
import json
import numpy as np
import onnxruntime as ort
from collections import deque

data_dir = "data/"
training_dir = "training_outputs/"

# --- CONFIGURATION ---
BROKER_IP = "192.168.2.110" # CHANGE TO YOUR PC'S IP
TOPIC = "S25U/#" # Standard topic for most MQTT sensor apps
CSV_FILENAME = data_dir + "gesture_dataset_raw.csv"
WINDOW_SIZE = 100

# --- LOAD EDGE MODEL ---
print("Loading ONNX model and scalers...")
ort_session = ort.InferenceSession(training_dir + "tiny_gesture.onnx")
scaler_mean = np.load(training_dir + "scaler_mean.npy")
scaler_scale = np.load(training_dir + "scaler_scale.npy")

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
                
                # 1. Convert to numpy and normalize using training scalers
                features = np.array(live_buffer)
                features = (features - scaler_mean) / scaler_scale
                
                # 2. Reshape for PyTorch/ONNX 1D-CNN format: (Batch, Channels, Sequence_Length)
                features = features.T.reshape(1, 6, WINDOW_SIZE).astype(np.float32)
                
                # 3. Run Edge Inference via ONNX
                ort_inputs = {ort_session.get_inputs()[0].name: features}
                ort_outs = ort_session.run(None, ort_inputs)
                
                # 4. Get Prediction
                prediction = np.argmax(ort_outs[0])
                
                if prediction != 0: # Only trigger if not "Idle"
                    print(f"\n🚀 [EVENT TRIGGERED]: {CLASSES[prediction]}")
                    print(f"   |-> Simulated SRAM Footprint: ~12.8 KB")
                    print(f"   |-> Computational Cost: ~38,400 MACs")
                    print(f"   |-> Flash Weight Size: < 8 KB")
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
    
    # Ensure this matches your Mosquitto broker setup
    client.connect(BROKER_IP, 1883, 60)
    
    # Blocking loop to keep the script running
    client.loop_forever()