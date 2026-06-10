# Driver Monitoring System (DMS) - Features & Metrics Guide

Welcome! This document explains how the **`integrated_dms.py`** script works in simple, everyday language. It breaks down what the system does, how it measures your behavior, and how it keeps you safe on the road.

---

## 📊 The Real-Time Dashboard
When the script runs, a dashboard overlay is displayed in the top-left corner of the camera feed. This acts as the control panel of the system, showing your current states and numbers at a glance.

The color of the dashboard changes dynamically based on your safety level:
*   🟢 **Green:** Safe and alert.
*   🟡 **Yellow/Orange:** Distracted or fatigued.
*   🔴 **Red:** Critical alerts (Drowsiness or Phone Usage).

Here are the key metrics shown on the dashboard:

### 1. `STATUS` (Current State)
This shows your overall driver status. It can be one of five states:
*   **`SAFE`**: You are looking forward and driving safely.
*   **`DROWSY`**: You have closed your eyes for too long.
*   **`FATIGUED`**: You are yawning.
*   **`PHONE_USAGE`**: You are holding or using your mobile phone.
*   **`DISTRACTED`**: You are looking down away from the road.

### 2. `EAR` (Eye Aspect Ratio)
*   **What it is:** A mathematical calculation that measures how open your eyes are.
*   **How it works:** The system maps points around your left eye and calculates the ratio of height to width. 
*   **The Threshold:** A normal open eye is usually around `0.30` or higher. If your EAR drops below **`0.20`**, the system registers that your eyes are closed.
*   **Alert Trigger:** If your EAR stays below `0.20` for **more than 2.0 seconds**, your status changes to `DROWSY`.

### 3. `MAR` (Mouth Aspect Ratio)
*   **What it is:** A measurement of how open your mouth is, used to detect yawning.
*   **How it works:** The system calculates the distance (in pixels) between your upper and lower lips.
*   **Alert Trigger:** If the distance (MAR) is **greater than 25**, the system detects a yawn and sets your status to `FATIGUED`.

### 4. `POSE` (Head Pose Orientation)
*   **What it is:** Tracks where you are looking.
*   **How it works:** The system uses 6 key points on your face (eyes, nose, mouth corners, chin) to calculate the 3D direction of your head.
*   **Directions detected:**
    *   **Looking Left** (turned left > 10 degrees)
    *   **Looking Right** (turned right > 10 degrees)
    *   **Looking Down** (tilted down > 10 degrees)
    *   **Forward** (safe baseline view)

### 5. `PHONE` (Mobile Usage Detection)
*   **What it is:** Detects whether you are holding or using a phone.
*   **How it works:** The AI model (YOLOv8) scans the video feed for mobile phones.
*   **Alert Trigger:** If a phone is detected continuously for **more than 2.0 seconds**, your status changes to `PHONE_USAGE`.

---

## 🎯 Driver Risk Score
The system tracks a **Driver Risk Score** to evaluate overall safety during a trip.

*   **Starting Point:** You start with a perfect score of **`100`**.
*   **Penalties:** If you violate a safety rule, points are immediately deducted from your score (once per event transition):
    *   🔴 **Drowsy:** `-30` points
    *   🔴 **Phone Usage:** `-20` points
    *   🟡 **Distracted:** `-15` points
    *   🟡 **Fatigued:** `-10` points
*   **Recovery:** The system encourages safe driving! If your status remains **`SAFE`** continuously for **5.0 seconds**, your score recovers by **`+1` point** (up to a maximum of `100`).

---

## 🚨 Alarm & Warning System
The script has different alarm responses for different severity levels. To prevent video lag, the alarm beeps run on a separate background thread.

| Alert State | Persistence Delay | Sound Alarm? | Screen Warning? |
| :--- | :--- | :--- | :--- |
| **DROWSY** | Eyes closed > 2 seconds | 🔊 **Immediate Beep** | 🖥️ **Yes** (Shows `Closed: X.Xs` timer) |
| **PHONE_USAGE** | Phone visible > 2 seconds | 🔊 **Beep after 2s** | 🖥️ **Yes** (Red box + `Phone: X.Xs` timer) |
| **DISTRACTED** | Looking down > 3 seconds | 🔊 **Beep after 3s** | 🖥️ **Yes** (Shows `Distracted: X.Xs` timer) |
| **FATIGUED** | Instantaneous yawn detection | 🔇 **No Sound** | 🖥️ **Yes** (Status changes to `FATIGUED`) |

---

## 📁 Automatic Event Logging
All critical safety events are recorded with real-time timestamps in a file named **`dms_events.log`** in your project folder. The system logs:
1.  When the camera feed starts and stops.
2.  When a face is detected or lost.
3.  When a driver status changes (e.g. `SAFE -> PHONE_USAGE`).
4.  When seatbelt status changes (e.g. `Not Detected -> Worn`).
5.  Score updates and recovery info.

---

## 🎥 Video Recording
*   The script automatically records your entire driving/testing session.
*   The output is saved as a video file named **`demo.mp4`** in your project folder.
*   This video includes all dashboard overlays, bounding boxes, and warning timers drawn in real-time, making it perfect for review or presentations.
