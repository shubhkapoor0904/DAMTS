# Implementation Plan - DAMTS Vercel Deployment

Transition the local Python OpenCV Driver Alertness Monitoring & Tracking System (DAMTS) into a Vercel-ready, client-side web application. The web app will replicate the exact system logic, thresholds, visual HUD overlays, audio alerts, log files, and session video recordings entirely in the browser using MediaPipe and ONNX Runtime Web.

---

## User Review Required

Because local terminal commands are blocked by system-level write permissions in the agent environment, you must execute two preparation scripts locally to generate the web-friendly YOLOv8 assets:

1. **Model Class Inspection**: Run `python inspect_model.py` to extract the class name map from `best2.pt`. This writes `model_info.json`.
2. **ONNX Export**: Run `python export_onnx.py` to export the PyTorch weights `best2.pt` to `best2.onnx`.

> [!IMPORTANT]
> The exported `best2.onnx` and `model_info.json` must remain in the project root directory. The web application will fetch them dynamically when loaded.

---

## Proposed Changes

We will create a pure, high-performance static web application in the root directory. This makes it instantly deployable to Vercel with zero build configuration.

### [NEW] Web Frontend Files

#### [NEW] [index.html](file:///d:/DAMTS_MediaPipe/index.html)
- Provides the HTML5 semantic layout.
- Integrates external CDNs:
  - MediaPipe Tasks-Vision (`@mediapipe/tasks-vision`) for FaceMesh.
  - ONNX Runtime Web (`ort.min.js`) for YOLOv8 inference.
  - Chart.js for real-time telemetry plotting.
  - Lucide Icons for premium visual UI indicators.
- Hosts the video element, the overlay canvas, controls (Start, Pause, Stop/Save), the system logs panel, and the telemetry dashboard.

#### [NEW] [styles.css](file:///d:/DAMTS_MediaPipe/styles.css)
- Premium dark-theme CSS with glassmorphic panels (`backdrop-filter: blur`).
- Custom animations, neon glow highlights (cyberpunk blue/cyan for normal states, red for alert states), and responsive layout grids.

#### [NEW] [app.js](file:///d:/DAMTS_MediaPipe/app.js)
Represents the core application controller:
- **Camera Stream**: Manages `navigator.mediaDevices.getUserMedia` feed.
- **MediaPipe FaceMesh**: Configures `FaceLandmarker` running locally in WebAssembly.
- **YOLOv8 Inference**:
  - Resizes input frame to `640x640`.
  - Normalizes pixel values and transposes shape to planar `[1, 3, 640, 640]`.
  - Performs inference using `onnxruntime-web`.
  - Parses outputs `[1, 4 + C, 8400]`, runs Non-Maximum Suppression (NMS), and maps bounding boxes for `"Set_belt"` and `"smartphone"` / `"Dist_mob"`.
- **Calibration Engine**: Replicates the 3-second eye calibration and 3-second head-pose calibration with identical mathematical thresholds.
- **Decision Engine**: Monitors EAR, MAR, Head Pose angles, and phone presence to compute status (`SAFE`, `DROWSY`, `PHONE_USAGE`, `FATIGUED`, `DISTRACTED`) and safety score (starting at 100, dynamic deductions, and +1/5s recovery).
- **Sound Alerts**: Synthesizes 2000Hz beeps using the Web Audio API (`OscillatorNode`).
- **Telemetry & Logging**:
  - Formats logs into a downloadable `dms_events.log`.
  - Compiles telemetry confidence into a downloadable `phone_confidence_telemetry.csv` and updates a Chart.js real-time graph.
- **Session Video Recording**: Leverages the `MediaRecorder` API to record the annotated `<canvas>` stream, creating a downloadable `demo.mp4` file upon session termination.

#### [NEW] [vercel.json](file:///d:/DAMTS_MediaPipe/vercel.json)
- Configures custom headers for Vercel deployment:
  - `Cross-Origin-Opener-Policy: same-origin`
  - `Cross-Origin-Embedder-Policy: require-corp`
- **Why?** These headers enable WebAssembly Multi-Threading via `SharedArrayBuffer` in modern browsers, speeding up YOLOv8 and FaceMesh inference by up to 10x.

---

## Verification Plan

### Automated Tests
Since the application runs client-side in the browser, testing can be performed using local hosting.

### Manual Verification
1. Run the inspection and export commands in your local shell:
   ```bash
   python inspect_model.py
   python export_onnx.py
   ```
2. Start a local server in the project directory:
   ```bash
   # Option A: python
   python -m http.server 8000
   
   # Option B: Node.js
   npx serve .
   ```
3. Open `http://localhost:8000` in Google Chrome or Microsoft Edge.
4. Verify:
   - Camera access works.
   - Calibration phases complete successfully.
   - Drowsiness, fatigue, looking down/away, seatbelt compliance, and phone usage detections trigger matching overlays and audio alerts.
   - Clicking "Stop & Save" generates downloads for `demo.mp4`, `dms_events.log`, and `phone_confidence_telemetry.csv`.
