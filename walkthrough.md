# DMS Calibration Step Walkthrough

We have successfully integrated a robust, visual-first calibration phase at startup in the Driver Monitoring System (DMS).

## Changes Made

### 1. State Variables & Accumulators
Defined tracking logic in `integrated_dms.py`:
- `calibration_state` starts as `CALIB_STATE_EYES`.
- `CALIBRATION_DURATION` is set to `3.0` seconds per phase.
- Lists `eye_ear_samples`, `pose_x_samples`, and `pose_y_samples` accumulate landmarks during the active calibration phases.
- `calib_elapsed_time` tracks time spent with a valid face detected, pausing if the face is lost.

### 2. Phase 1: Open-Eye Calibration
- Runs for 3 seconds of valid face tracking.
- Prompts the user to look straight ahead with open eyes.
- Collects EAR values, computes the average, and dynamically sets the drowsiness threshold:
  $$\text{EYE\_THRESHOLD} = \text{clip}(\text{avg\_ear} \times 0.70, 0.15, 0.25)$$
- Logs progress and final threshold settings to standard logs.

### 3. Phase 2: Head-Pose Calibration
- Runs for 3 seconds of valid face tracking immediately after Phase 1.
- Prompts the user to sit in their natural driving posture.
- Computes baseline pitch and yaw rotation angles:
  $$\text{baseline\_x\_angle} = \text{avg}(x\_samples)$$
  $$\text{baseline\_y\_angle} = \text{avg}(y\_samples)$$
- Logs final baseline angles.

### 4. Normalized Runtime Detection
During normal monitoring (`STATE_MONITORING` / `CALIB_STATE_DONE`):
- Yaw and pitch angles are adjusted relative to the baseline:
  $$\text{adjusted\_x} = \text{x\_angle} - \text{baseline\_x\_angle}$$
  $$\text{adjusted\_y} = \text{y\_angle} - \text{baseline\_y\_angle}$$
- Pose triggers Left/Right/Down looking alerts based on these baseline-relative angles.
- EAR is evaluated against the dynamically computed dynamic threshold.

### 5. Premium UI HUD Overlay
Implemented a translucent (glassmorphic) card at the bottom of the video frame during calibration:
- Dark slate fill with neon blue/cyan glowing border.
- Dynamic instruction texts:
  - **Step 1**: `"PHASE 1: EYE CALIBRATION - Look straight ahead and keep your eyes open"`
  - **Step 2**: `"PHASE 2: HEAD POSE CALIBRATION - Maintain normal driving posture, look straight ahead"`
- Real-time feedback status that detects face availability:
  - **Green**: `"Status: Calibrating (X.Xs left)"` when tracking is active.
  - **Red**: `"Status: NO FACE DETECTED (Calibration Paused)"` when the face is lost.
- A glowing progress bar (green when calibrating, orange when paused) that dynamically fills based on calibration completion.

---

## How to Run & Verify

1. Run the script:
   ```bash
   python integrated_dms.py
   ```
2. **Phase 1 (Eyes Open)**: Align your face and look straight at the webcam. The progress bar will fill in green. If you move out of frame, it will turn red/orange and pause.
3. **Phase 2 (Driving Posture)**: Keep looking straight ahead in your default driving posture as the bar fills again.
4. **Active Monitoring**: Once complete, the card disappears, and the normal monitoring dashboard starts.
5. Check your `dms_events.log` file for output details similar to:
   ```text
   2026-06-13 01:05:00 - INFO - Open-eye calibration complete. Average EAR: 0.3120. Dynamic EYE_THRESHOLD set to 0.2184
   2026-06-13 01:05:03 - INFO - Head-pose calibration complete. Baseline X: -2.31, Baseline Y: 1.45
   ```
