import paho.mqtt.client as mqtt
import json
import pandas as pd
import keyboard
import time

data_dir = "data/"

# --- CONFIGURATION ---
BROKER_IP = "192.168.2.110" # CHANGE TO YOUR PC'S IP
TOPIC = "S25U/#" # Standard topic for most MQTT sensor apps
CSV_FILENAME = data_dir + "gesture_dataset_raw.csv"

# Global dictionary to hold the latest readings to sync Accel + Gyro
current_state = {
    "timestamp": 0,
    "ax": 0.0, "ay": 0.0, "az": 0.0,
    "gx": 0.0, "gy": 0.0, "gz": 0.0
}

data_buffer = []

def on_connect(client, userdata, flags, rc):
    print(f"Connected to Mosquitto Broker with code {rc}")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global current_state, data_buffer
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        sensor_type = payload.get("type", "").lower()
        values = payload.get("values", [0,0,0,0,0,0])
        
        # Update the rolling state (grabbing only the first 3 raw values)
        if "accelerometer" in sensor_type:
            current_state["ax"], current_state["ay"], current_state["az"] = values[:3]
            current_state["timestamp"] = payload.get("timestamp", time.time_ns())
        elif "gyroscope" in sensor_type:
            current_state["gx"], current_state["gy"], current_state["gz"] = values[:3]
            
        # Determine the current label via Keyboard input
        current_label = 0 # Default: Idle / Null class
        if keyboard.is_pressed('1'):
            current_label = 1 # Gesture 1
        elif keyboard.is_pressed('2'):
            current_label = 2 # Gesture 2
            
        # Append only if we just updated the accelerometer (acts as our 100Hz clock ticker)
        if "accelerometer" in sensor_type:
            data_buffer.append([
                current_state["timestamp"], 
                current_state["ax"], current_state["ay"], current_state["az"],
                current_state["gx"], current_state["gy"], current_state["gz"],
                current_label
            ])
            
    except Exception as e:
        print(f"Error parsing message: {e}") # Print the error instead of silently failing

# --- SETUP MQTT CLIENT ---
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_IP, 1883, 60)

print("Starting data ingestion...")
print("Hold '1' for Gesture 1 | Hold '2' for Gesture 2")
print("Press 'q' to stop and save data.")

client.loop_start()

# --- MAIN LOOP ---
while True:
    if keyboard.is_pressed('q'):
        print("\nStopping and saving to CSV...")
        break
    time.sleep(0.1)

client.loop_stop()

# --- SAVE DATA ---
columns = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label"]
df = pd.DataFrame(data_buffer, columns=columns)
df.to_csv(CSV_FILENAME, index=False)
print(f"Saved {len(df)} samples to {CSV_FILENAME}")