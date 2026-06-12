# DAMTS: Driver Alertness Monitoring & Tracking System

An integrated, real-time Driver Monitoring System (DMS) combining **MediaPipe FaceMesh** for landmark tracking and a custom **YOLOv8** model for object detection. DAMTS monitors driver state, evaluates risk metrics, tracks safety violations (e.g., seatbelt non-compliance, phone usage, distraction, and drowsiness), manages a dynamic safety score, and triggers platform-independent audio alerts.

---

## 🚀 Key Features

- **Drowsiness Detection**: Uses **Eye Aspect Ratio (EAR)** calculated from MediaPipe landmarks to identify closed eyes. Alerts trigger when eyes are closed for more than **2 seconds**.
- **Fatigue / Yawn Detection**: Measures **Mouth Aspect Ratio (MAR)** to identify yawning.
- **Head Pose & Distraction Tracking**: Leverages 3D-to-2D projection and `solvePnP` on key facial landmarks to determine pitch and yaw. Triggers alerts when the driver is looking away (e.g., looking down at a phone/dashboard) for more than **3 seconds**.
- **Object Detection (Custom YOLOv8)**: Real-time detection of:
  - Mobile phone usage (`Dist_mob`)
  - Seatbelt compliance (`Set_belt`)
- **Dynamic Driver Score**:
  - The driver starts with a score of **100**.
  - Deductions are applied for unsafe behaviors (e.g., Drowsiness: `-30`, Phone Usage: `-20`, Distraction: `-15`, Fatigue: `-10`).
  - Gradual recovery (+1 point every 5 seconds) occurs when the driver returns to a **SAFE** state.
- **Robust Audio Alert Engine**: Triggers platform-specific beeps/alarms across Windows, macOS, and Linux to keep the driver alert.
- **Logging & Video Recording**: Saves event telemetry in `dms_events.log` and records the overlayed session to `demo.mp4`.
- **Heads-Up Dashboard (HUD)**: Overlays real-time metrics (Status, EAR, MAR, Pose, Phone detection status, Driver Score, Seatbelt status, and Alert timers) directly onto the video feed.

---

## 📁 Repository Structure

```
DAMTS/
├── integrated_dms.py        # Main execution script integrating MediaPipe and YOLOv8
├── best.pt                  # Pre-trained YOLOv8 weights (detecting mobile usage and seatbelts)
├── dms_events.log           # Automatically generated execution/event logs
├── demo.mp4                 # Saved output video recording of the run
├── experiments/             # Experimental scripts testing individual features
│   ├── ear_test.py          # Isolated Eye Aspect Ratio testing
│   ├── yawn_detection.py    # Isolated Yawn detection using Mouth Aspect Ratio
│   ├── head_pose.py         # Head pose estimation using solvePnP
│   ├── test_yolo.py         # YOLO model verification script
│   └── ...
└── notebooks/               # Training notebooks
    └── train-yolov8-object-detection-on-custom-dataset.ipynb  # Custom dataset training guide
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/shubhkapoor0904/DAMTS.git
cd DAMTS
```

### 2. Install Dependencies
Make sure you have Python (version >= 3.8 recommended) installed. Install the required Python packages:
```bash
pip install opencv-python mediapipe numpy ultralytics
```
*Note: For Windows users, the system uses the native `winsound` library. For macOS and Linux users, appropriate fallback mechanisms are implemented (e.g., using `osascript`, `paplay`, `pw-play`, `aplay`, or standard system terminal alerts).*

### 3. Place YOLOv8 Weights
Ensure that the trained model file `best.pt` is present in the root directory.

---

## 💻 Usage

To run the integrated Driver Monitoring System using your default webcam:
```bash
python integrated_dms.py
```

### Controls:
- **`ESC` key**: Safely stop the application, release the camera, and save the session recording to `demo.mp4`.

---

## 🧠 System Logic & Thresholds

| Behavior | Logic / Measurement | Trigger Condition | Penalty | Action / Alert |
| :--- | :--- | :--- | :--- | :--- |
| **Drowsiness** | Eye Aspect Ratio (EAR) < `0.20` | > 2.0 seconds | -30 | Audio Alert + WARNING |
| **Phone Usage** | Custom YOLOv8 detection of `Dist_mob` | > 2.0 seconds | -20 | Audio Alert + WARNING |
| **Distraction** | Head pitch/yaw rotation angles | Look down > 3.0 seconds | -15 | Audio Alert + WARNING |
| **Fatigue (Yawning)**| Mouth Aspect Ratio (MAR) > `25` | Instantly | -10 | Status HUD Update |
| **Seatbelt Status** | Custom YOLOv8 detection of `Set_belt` | Instantly | None | Status HUD Update |

### Score Recovery
When the driver returns to a **SAFE** state, the score recovers by **+1 point** every **5 seconds**, up to a maximum score of **100**.

---

## 📊 Telemetry and Logs
All system transitions, penalties, and sensor status updates are automatically logged with timestamps in `dms_events.log`. For example:
```text
2026-06-12 17:00:00 - INFO - DMS System initialized. Starting video capture...
2026-06-12 17:00:05 - INFO - Face detected.
2026-06-12 17:00:10 - WARNING - Status changed: -> PHONE_USAGE
2026-06-12 17:00:10 - INFO - Penalty applied: PHONE_USAGE (-20). Score: 80
2026-06-12 17:00:15 - INFO - Status changed: -> SAFE
2026-06-12 17:00:20 - INFO - Score recovered: +1. Score: 81
```
