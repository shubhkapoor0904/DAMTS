# Automatic Calibration Steps for DMS

This plan details the addition of an automatic calibration phase at the startup of the Driver Monitoring System (DMS) before normal monitoring begins.

## User Review Required

> [!IMPORTANT]
> **Dynamic EAR Threshold Formula**: We propose setting the dynamic eye threshold as `EYE_THRESHOLD = clip(average_ear * 0.70, 0.15, 0.25)`. A value of `0.70` (70% of the driver's normal open-eye average EAR) is standard to prevent false positives while ensuring reliable drowsiness detection. The clipping bounds (`0.15` and `0.25`) act as fallbacks in case the user blinks heavily or is misaligned during calibration.
> Please review and confirm if this formula aligns with your expectations.

## Proposed Changes

### DMS Main Script

#### [MODIFY] [integrated_dms.py](file:///d:/DAMTS_MediaPipe/integrated_dms.py)

- **Introduce Calibration States**:
  - `STATE_CALIBRATING_EYES`: First phase (3 seconds) to capture average EAR.
  - `STATE_CALIBRATING_POSE`: Second phase (3 seconds) to capture average X and Y head pose angles.
  - `STATE_MONITORING`: The final active monitoring state.

- **Data Accumulation**:
  - Store EAR and head-pose angles in lists during their respective calibration phases.
  - Only capture data and decrement timers when a face is detected (`face_detected_this_frame` is `True`).

- **Dynamic Calibration Logic**:
  - **Open-Eye Phase**: Calculate `avg_ear = sum(ear_samples) / len(ear_samples)` and set `EYE_THRESHOLD = clip(avg_ear * 0.70, 0.15, 0.25)`.
  - **Head-Pose Phase**: Calculate `baseline_x = sum(x_samples) / len(x_samples)` and `baseline_y = sum(y_samples) / len(y_samples)`.
  - **Runtime Normalization**: Adjust runtime angles as `adjusted_x = x_angle - baseline_x` and `adjusted_y = y_angle - baseline_y`. Apply threshold checks (`<-10` and `>10`) relative to these adjusted values.

- **Aesthetic On-Screen Calibration Overlay**:
  - Overlay a semi-transparent card (glassmorphism look) for instructions using `cv2.addWeighted`.
  - Show a countdown progress bar (neon blue/teal color) dynamically filling up based on time remaining.
  - Display user-friendly status updates:
    - `"CALIBRATION: PHASE 1/2 (EYES OPEN)"`
    - `"CALIBRATION: PHASE 2/2 (DRIVING POSTURE)"`
    - Status check indicators: `"Collecting data..."` or `"No face detected (Paused)"`.

## Verification Plan

### Automated Tests
- Since this is a live webcam app, automated unit tests aren't directly applicable for end-to-end flow. However, we will ensure that:
  - The script compiles and starts without issues.
  - Calibration state advances only when a face is present.
  - Calculated baselines and dynamic thresholds are printed to logs upon transition.

### Manual Verification
1. Start the script via `python integrated_dms.py`.
2. Keep eyes open and look at the camera for Phase 1. Confirm that calibration timer decrements and dynamic EAR threshold is set.
3. Keep natural driving posture for Phase 2. Confirm baseline pitch/yaw are captured.
4. Move head left, right, and down to verify that detection relies on the relative/adjusted angles.
5. Inspect `dms_events.log` to check calibration output.
