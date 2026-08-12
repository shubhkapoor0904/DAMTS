# DAMTS Vercel Deployment Walkthrough

We have successfully migrated the local Python OpenCV Driver Alertness Monitoring & Tracking System (DAMTS) into a Vercel-ready, client-side web application. This layout replicates 100% of the original calibration logic, decision thresholds, safety scoring, warning alarm sound, and event log file exporters.

---

## 📁 Created Files

1. **[index.html](file:///d:/DAMTS_MediaPipe/index.html)**: The HTML5 web application dashboard layout. Loads MediaPipe FaceMesh, ONNX Runtime Web, Chart.js, and Lucide Icons via high-speed CDNs.
2. **[styles.css](file:///d:/DAMTS_MediaPipe/styles.css)**: A premium dark-theme stylesheet with responsive grid alignment, custom control button hovers, neon status highlights, and card layouts.
3. **[app.js](file:///d:/DAMTS_MediaPipe/app.js)**: The core client-side controller. Manages user webcam frames, runs FaceLandmarker & custom YOLOv8 model inference, controls calibration timers, calculates safety score metrics, triggers browser audio, updates the live telemetry chart, logs console warnings, and records the canvas stream to output a `demo.mp4` session download.
4. **[vercel.json](file:///d:/DAMTS_MediaPipe/vercel.json)**: The Vercel configuration setting `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers, enabling browser multi-threaded WebAssembly execution.
5. **[inspect_model.py](file:///d:/DAMTS_MediaPipe/inspect_model.py)**: Python script to extract class names from `best2.pt` and write them to `model_info.json`.
6. **[export_onnx.py](file:///d:/DAMTS_MediaPipe/export_onnx.py)**: Python script to export PyTorch weights `best2.pt` to standard ONNX `best2.onnx` format.

---

## 🚀 Execution & Verification Steps

### Step 1: Export YOLO Model & Metadata
Run the following commands in your local workspace terminal:

```bash
# 1. Generate model_info.json containing class name configurations
python inspect_model.py

# 2. Export best2.pt weights to best2.onnx (will take ~10-30 seconds)
python export_onnx.py
```

This will produce `model_info.json` and `best2.onnx` directly in your workspace root directory. The web page loads these files locally.

### Step 2: Start a Local Server
Start a lightweight web server in your workspace folder. You can use either Node.js or Python:

```bash
# Option A: Using Python (built-in)
python -m http.server 8000

# Option B: Using Node.js
npx serve .
```

### Step 3: Test the Application
1. Open your web browser and navigate to `http://localhost:8000`.
2. Allow webcam permissions when prompted.
3. Click **Start Session**.
4. **Phase 1 Calibration (Eyes Open)**: Keep looking straight ahead at the camera. Watch the green progress bar at the bottom fill up. If you look away, the bar turns orange and pauses.
5. **Phase 2 Calibration (Head Pose)**: Sit in your normal driving posture and look straight. The bar fills again to calibrate your baseline.
6. **Monitoring Active**: Once complete, the calibration HUD disappears. You will see the green telemetry dashboard on the top-left overlay.
7. Test alerts:
   - **Drowsiness**: Close your eyes for more than 2 seconds. The HUD will flash red and play a warning beep.
   - **Phone Usage**: Raise a phone in front of the camera. A red bounding box will appear around it, and the HUD will report `PHONE_USAGE` and trigger warning sounds.
   - **Distraction**: Turn your head left, right, or look down. The HUD will detect the rotation relative to your calibrated baseline and trigger alerts when looking away for more than 3 seconds.
   - **Fatigue**: Yawn wide. The HUD will report `FATIGUED`.
   - **Seatbelt**: Ensure your custom model registers seatbelts. It will draw a green bounding box and mark `SEATBELT: Worn` in the dashboard.
8. Click **Stop & Save Session**.
   - Your browser will automatically compile and download:
     - `demo.mp4` (containing the video recording with all HUD dashboards, text overlays, and bounding boxes rendered directly on the frames).
     - `dms_events.log` (containing a text file of all chronological status transitions, calibration values, and warning events).
     - `phone_confidence_telemetry.csv` (containing the time-series logs of phone detection confidence and box dimensions).

---

## ☁️ Deployment to Vercel

To deploy this site online, you can use the Vercel Git integration or the CLI:

### Option A: Vercel Dashboard (Easiest)
1. Push this folder to a GitHub, GitLab, or Bitbucket repository.
2. Log in to [Vercel](https://vercel.com) and click **Add New > Project**.
3. Select your repository.
4. Leave all settings at their default values (Vercel automatically detects the static project structure).
5. Click **Deploy**.

### Option B: Vercel CLI
If you have Vercel CLI installed:
```bash
# Login if not already logged in
vercel login

# Run deployment (instantly uploads all HTML/CSS/JS and model weights)
vercel --prod
```

> [!IMPORTANT]
> The included `vercel.json` file ensures that the site is served with the correct headers, enabling the high-performance multithreaded WASM runtime inside the browser.
