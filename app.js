import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8";

// ------------------------------------
// UI Elements & State Declarations
// ------------------------------------
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const loadingScreen = document.getElementById('loading-screen');
const loadingStatus = document.getElementById('loading-status');
const loadingSubstatus = document.getElementById('loading-substatus');

const btnStart = document.getElementById('btn-start');
const btnPause = document.getElementById('btn-pause');
const btnStop = document.getElementById('btn-stop');
const btnMute = document.getElementById('btn-mute');
const btnDownloadLogs = document.getElementById('btn-download-logs');
const btnDownloadTelemetry = document.getElementById('btn-download-telemetry');
const recIndicator = document.getElementById('rec-indicator');

const hudScore = document.getElementById('hud-score');
const hudScoreBar = document.getElementById('hud-score-bar');
const hudState = document.getElementById('hud-state');
const hudEar = document.getElementById('hud-ear');
const hudMar = document.getElementById('hud-mar');
const hudPose = document.getElementById('hud-pose');
const hudPhone = document.getElementById('hud-phone');
const hudSeatbelt = document.getElementById('hud-seatbelt');

// App Variables
let faceLandmarker = null;
let yoloSession = null;
let classNames = { 0: "Set_belt", 1: "smartphone" }; // standard fallback
let isRunning = false;
let isMuted = false;
let lastFrameTime = null;
let animationFrameId = null;

// Recording & Download Buffers
let mediaRecorder = null;
let recordedChunks = [];
let systemLogs = [];
let telemetryData = [];
let telemetryChart = null;

// Calibration State and Configuration
const CALIB_STATE_EYES = "EYES";
const CALIB_STATE_POSE = "POSE";
const CALIB_STATE_DONE = "DONE";
const CALIBRATION_DURATION = 3.0; // 3 seconds per phase

let calibrationState = CALIB_STATE_EYES;
let calibElapsedTime = 0.0;
let lastValidFrameTime = null;

let eyeEarSamples = [];
let poseXSamples = [];
let poseYSamples = [];

let EYE_THRESHOLD = 0.20;
let baselineXAngle = 0.0;
let baselineYAngle = 0.0;

// Driver Monitor Variables
let closedStart = null;
let phoneAccumulatedTime = 0.0;
let distractedStartTime = null;
let driverScore = 100;
let safeStartTime = null;

// Alarm State
let alarmPlaying = false;
let audioCtx = null;

// YOLO Configuration
const YOLO_CONF_BASE = 0.2;
const YOLO_CONF_PHONE_CONFIRMED = 0.5;
const YOLO_CONF_SEATBELT = 0.25;
const PHONE_DECAY_RATE = 1.0;
const PHONE_MAX_ACCUMULATION = 2.5;

// Global detection states for drawing
let activeDetections = [];
let faceDetectedThisFrame = false;

// ------------------------------------
// Auxiliary Utilities (Logging, Sound, Chart)
// ------------------------------------

function logEvent(message, level = "INFO") {
  const now = new Date();
  const dateStr = now.getFullYear() + '-' + 
    String(now.getMonth() + 1).padStart(2, '0') + '-' + 
    String(now.getDate()).padStart(2, '0') + ' ' + 
    String(now.getHours()).padStart(2, '0') + ':' + 
    String(now.getMinutes()).padStart(2, '0') + ':' + 
    String(now.getSeconds()).padStart(2, '0');
  
  const logLine = `${dateStr} - ${level} - ${message}`;
  systemLogs.push(logLine);
  
  // Append to onscreen console
  const consoleEl = document.getElementById("log-console");
  if (consoleEl) {
    const entry = document.createElement("div");
    entry.className = `log-entry log-${level.toLowerCase()}`;
    entry.textContent = `[${level}] ${logLine.substring(22)}`;
    consoleEl.appendChild(entry);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
  
  if (level === "ERROR") console.error(logLine);
  else if (level === "WARNING") console.warn(logLine);
  else console.log(logLine);
}

function playBeepSound(frequency = 2000, duration = 400) {
  if (isMuted) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = 'sine';
    osc.frequency.value = frequency;
    
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration / 1000);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + duration / 1000);
  } catch (err) {
    console.error("Audio Alert generation failed: ", err);
  }
}

function triggerAudioAlarm() {
  if (!alarmPlaying) {
    alarmPlaying = true;
    playBeepSound(2000, 400);
    setTimeout(() => {
      alarmPlaying = false;
    }, 450);
  }
}

function initChart() {
  const chartCanvas = document.getElementById('telemetry-chart');
  telemetryChart = new Chart(chartCanvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Phone Confidence',
        data: [],
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        tension: 0.2,
        fill: true,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { 
          min: 0, 
          max: 1.0, 
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#9ca3af' }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function updateChart(confidence) {
  if (!telemetryChart) return;
  const labels = telemetryChart.data.labels;
  const data = telemetryChart.data.datasets[0].data;
  
  labels.push('');
  data.push(confidence);
  
  if (data.length > 50) {
    labels.shift();
    data.shift();
  }
  telemetryChart.update('none');
}

// ------------------------------------
// System Initialization
// ------------------------------------

async function initializeApp() {
  try {
    // 1. Load MediaPipe FaceLandmarker
    loadingStatus.textContent = "Loading FaceMesh model...";
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 1
    });
    logEvent("MediaPipe FaceLandmarker initialized successfully.", "INFO");

    // 2. Fetch model metadata if available
    loadingStatus.textContent = "Checking YOLO configuration...";
    try {
      const response = await fetch('./model_info.json');
      if (response.ok) {
        const metadata = await response.json();
        if (metadata && metadata.names) {
          classNames = metadata.names;
          logEvent(`Loaded class mapping from metadata: ${JSON.stringify(classNames)}`, "INFO");
        }
      } else {
        logEvent("No model_info.json metadata found. Using default coco/custom mapping.", "WARNING");
      }
    } catch (e) {
      logEvent("Failed to load model_info.json metadata. Using default coco mapping.", "WARNING");
    }

    // 3. Load ONNX Runtime YOLOv8 model
    loadingStatus.textContent = "Loading YOLOv8 ONNX weights...";
    loadingSubstatus.textContent = "Downloading best2.onnx (approx 5.5MB)...";
    try {
      yoloSession = await ort.InferenceSession.create('./best2.onnx', {
        executionProviders: ['wasm'],
        numThreads: 4
      });
      logEvent("ONNX Runtime YOLOv8 model loaded successfully.", "INFO");
    } catch (e) {
      logEvent("Could not load best2.onnx. Phone and seatbelt detection will be unavailable.", "WARNING");
      logEvent(`ONNX error details: ${e.message}`, "WARNING");
    }

    // Initialize Chart
    initChart();

    // 4. Reveal controls
    loadingScreen.style.opacity = 0;
    setTimeout(() => {
      loadingScreen.style.display = 'none';
    }, 500);

    btnStart.removeAttribute('disabled');
    logEvent("DMS System initialized. Ready to start capture.", "INFO");
  } catch (error) {
    loadingStatus.textContent = "System Initialization Failed";
    loadingSubstatus.textContent = error.message;
    logEvent(`Initialization Error: ${error.message}`, "ERROR");
  }
}

// ------------------------------------
// YOLOv8 Inference Pipeline
// ------------------------------------

// Temporary offscreen canvas for resizing
const resizeCanvas = document.createElement('canvas');
resizeCanvas.width = 640;
resizeCanvas.height = 640;
const resizeCtx = resizeCanvas.getContext('2d');

async function runYOLO(videoFrame) {
  if (!yoloSession) return [];

  // Preprocess: draw to 640x640 canvas
  resizeCtx.drawImage(videoFrame, 0, 0, 640, 640);
  const imgData = resizeCtx.getImageData(0, 0, 640, 640);
  const data = imgData.data;

  // Planar shape [1, 3, 640, 640]
  const numPixels = 640 * 640;
  const inputData = new Float32Array(3 * numPixels);
  for (let i = 0; i < numPixels; i++) {
    inputData[i] = data[i * 4] / 255.0;               // R
    inputData[numPixels + i] = data[i * 4 + 1] / 255.0; // G
    inputData[2 * numPixels + i] = data[i * 4 + 2] / 255.0; // B
  }

  const tensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
  
  try {
    const outputs = await yoloSession.run({ [yoloSession.inputNames[0]]: tensor });
    const outputName = yoloSession.outputNames[0];
    const outputTensor = outputs[outputName];
    
    // Parsing output format [1, 4 + classes_count, 8400]
    const dims = outputTensor.dims;
    const numChannels = dims[1];
    const numAnchors = dims[2];
    const outputData = outputTensor.data;

    let candidates = [];
    const classCount = numChannels - 4;

    for (let col = 0; col < numAnchors; col++) {
      let maxScore = -1;
      let maxClassIdx = -1;
      for (let cl = 0; cl < classCount; cl++) {
        const score = outputData[(4 + cl) * numAnchors + col];
        if (score > maxScore) {
          maxScore = score;
          maxClassIdx = cl;
        }
      }

      if (maxScore >= YOLO_CONF_BASE) {
        const xc = outputData[0 * numAnchors + col];
        const yc = outputData[1 * numAnchors + col];
        const w = outputData[2 * numAnchors + col];
        const h = outputData[3 * numAnchors + col];
        
        // Convert to absolute boundary coordinates relative to 640x640 space
        const x1 = xc - w / 2;
        const y1 = yc - h / 2;
        const x2 = xc + w / 2;
        const y2 = yc + h / 2;

        candidates.push({
          box: [x1, y1, x2, y2],
          score: maxScore,
          classId: maxClassIdx,
          label: classNames[maxClassIdx] || `Class_${maxClassIdx}`
        });
      }
    }

    // Run NMS filtering
    return nonMaximumSuppression(candidates, 0.45);
  } catch (err) {
    console.error("YOLO inference failed:", err);
    return [];
  }
}

function nonMaximumSuppression(boxes, iouThreshold) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  const active = new Array(boxes.length).fill(true);

  for (let i = 0; i < boxes.length; i++) {
    if (active[i]) {
      keep.push(boxes[i]);
      for (let j = i + 1; j < boxes.length; j++) {
        if (active[j]) {
          if (calculateIoU(boxes[i].box, boxes[j].box) > iouThreshold) {
            active[j] = false;
          }
        }
      }
    }
  }
  return keep;
}

function calculateIoU(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  if (interArea === 0) return 0.0;

  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  return interArea / (areaA + areaB - interArea);
}

// ------------------------------------
// Head Pose Calculations (solvePnP)
// ------------------------------------

function distance(p1, p2) {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
}

function computeHeadPose(landmarks, w, h) {
  // If OpenCV.js is not loaded, use a robust algebraic approximation
  if (typeof cv === 'undefined' || !cv.solvePnP) {
    const nose = landmarks[1];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeDist = Math.abs(rightEye.x - leftEye.x);
    const rawYaw = (nose.x - eyeMidX) / eyeDist;
    const y_angle = rawYaw * 120; // Replicates yaw degrees
    
    const chin = landmarks[199];
    const forehead = landmarks[10];
    const noseY = nose.y;
    const faceHeight = Math.abs(chin.y - forehead.y);
    const eyeMidY = (leftEye.y + rightEye.y) / 2;
    const upperDist = Math.abs(noseY - eyeMidY);
    const rawPitch = (upperDist / faceHeight) - 0.22;
    const x_angle = rawPitch * 150; // Replicates pitch degrees
    
    return { x_angle, y_angle };
  }

  try {
    const indices = [1, 33, 61, 199, 263, 291];
    let imagePoints = [];
    let objectPoints = [];

    indices.forEach(idx => {
      const lm = landmarks[idx];
      const px = lm.x * w;
      const py = lm.y * h;
      imagePoints.push(px, py);
      // Replicate flat-relative face geometry
      objectPoints.push(px, py, lm.z * w);
    });

    let imMat = cv.matFromArray(6, 1, cv.CV_64FC2, imagePoints);
    let objMat = cv.matFromArray(6, 1, cv.CV_64FC3, objectPoints);

    let focalLength = w;
    let camMat = cv.matFromArray(3, 3, cv.CV_64FC1, [
      focalLength, 0, w / 2,
      0, focalLength, h / 2,
      0, 0, 1
    ]);

    let distMat = cv.matFromArray(4, 1, cv.CV_64FC1, [0, 0, 0, 0]);

    let rvec = new cv.Mat();
    let tvec = new cv.Mat();

    let success = cv.solvePnP(objMat, imMat, camMat, distMat, rvec, tvec);
    let x_angle = 0.0;
    let y_angle = 0.0;

    if (success) {
      let rmat = new cv.Mat();
      cv.Rodrigues(rvec, rmat);
      
      let r = rmat.data64F;
      // Replicate RQDecomp3x3 angle extraction
      let pitchVal = Math.asin(-r[5]) * (180 / Math.PI);
      let yawVal = Math.atan2(r[2], r[8]) * (180 / Math.PI);
      
      x_angle = pitchVal * 360;
      y_angle = yawVal * 360;
      
      rmat.delete();
    }

    rvec.delete();
    tvec.delete();
    imMat.delete();
    objMat.delete();
    camMat.delete();
    distMat.delete();

    return { x_angle, y_angle };
  } catch (err) {
    // Fallback if matrix math crashes
    const nose = landmarks[1];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeDist = Math.abs(rightEye.x - leftEye.x);
    const rawYaw = (nose.x - eyeMidX) / eyeDist;
    const y_angle = rawYaw * 120;
    const chin = landmarks[199];
    const forehead = landmarks[10];
    const noseY = nose.y;
    const faceHeight = Math.abs(chin.y - forehead.y);
    const eyeMidY = (leftEye.y + rightEye.y) / 2;
    const upperDist = Math.abs(noseY - eyeMidY);
    const rawPitch = (upperDist / faceHeight) - 0.22;
    const x_angle = rawPitch * 150;
    return { x_angle, y_angle };
  }
}

// ------------------------------------
// Frame Processor Loop
// ------------------------------------

async function captureLoop() {
  if (!isRunning) return;

  const w = canvas.width;
  const h = canvas.height;

  // 1. Draw camera feed mirrored to match OpenCV layout
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();

  const currentTime = Date.now() / 1000;
  const dt = lastFrameTime ? (currentTime - lastFrameTime) : 0.0;
  lastFrameTime = currentTime;

  // Default detections
  let phoneDetected = false;
  let seatbeltDetected = false;
  let phoneConf = 0.0;

  // 2. Perform object detection (Only run YOLO if calibration completed)
  if (calibrationState === CALIB_STATE_DONE) {
    activeDetections = await runYOLO(video);
    
    activeDetections.forEach(det => {
      const [x1, y1, x2, y2] = det.box;
      
      // Rescale coordinates from 640x640 size back to output canvas size
      const scaleX = w / 640;
      const scaleY = h / 640;
      const rx1 = x1 * scaleX;
      const ry1 = y1 * scaleY;
      const rx2 = x2 * scaleX;
      const ry2 = y2 * scaleY;

      // Class mappings check
      if (det.label === "smartphone" || det.label === "Dist_mob") {
        phoneConf = det.score;
        const box_w = rx2 - rx1;
        const box_h = ry2 - ry1;
        const is_conf = phoneConf >= YOLO_CONF_PHONE_CONFIRMED ? 1 : 0;
        
        telemetryData.push({
          timestamp: currentTime,
          confidence: phoneConf,
          width: box_w,
          height: box_h,
          confirmed: is_conf
        });

        if (phoneConf >= YOLO_CONF_PHONE_CONFIRMED) {
          phoneDetected = true;
          // Draw Red Box (Mirrored to align with mirrored view)
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 3;
          ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
          ctx.fillStyle = '#ef4444';
          ctx.font = 'bold 14px sans-serif';
          ctx.fillText(`Phone: ${phoneConf.toFixed(2)}`, rx1, ry1 - 10);
        }
      } else if (det.label === "Set_belt" && det.score >= YOLO_CONF_SEATBELT) {
        seatbeltDetected = true;
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
        ctx.fillStyle = '#10b981';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`Seatbelt: ${det.score.toFixed(2)}`, rx1, ry1 - 10);
      }
    });

    updateChart(phoneConf);

    // Phone Persistence / Decay calculation
    if (phoneDetected) {
      phoneAccumulatedTime = Math.min(PHONE_MAX_ACCUMULATION, phoneAccumulatedTime + dt);
    } else {
      phoneAccumulatedTime = Math.max(0.0, phoneAccumulatedTime - PHONE_DECAY_RATE * dt);
    }
  }

  // 3. Process FaceMesh
  let faceResults = faceLandmarker.detectForVideo(video, Date.now());
  faceDetectedThisFrame = faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0;

  // Variables to populate HUD overlays
  let status = "SAFE";
  let pose = "Forward";
  let drowsy = false;
  let yawning = false;
  let phoneUsageAlert = false;
  let distractedAlert = false;
  let ear = 0.0;
  let mar = 0.0;
  let x_angle = 0.0;
  let y_angle = 0.0;
  let phoneDuration = phoneAccumulatedTime;
  let distractedDuration = 0.0;

  if (faceDetectedThisFrame) {
    const landmarks = faceResults.faceLandmarks[0];

    // EAR Drowsiness logic
    const leftEyeIndices = [33, 160, 158, 133, 153, 144];
    const eyePoints = leftEyeIndices.map(idx => {
      // Mirror landmark coordinate since drawing is flipped
      return [landmarks[idx].x * w, landmarks[idx].y * h];
    });

    // Draw eye landmark circles
    ctx.fillStyle = '#10b981';
    eyePoints.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    const v1 = distance(eyePoints[1], eyePoints[5]);
    const v2 = distance(eyePoints[2], eyePoints[4]);
    const hor = distance(eyePoints[0], eyePoints[3]);
    ear = (v1 + v2) / (2.0 * hor);

    // Mouth Aspect Ratio logic
    const upper = [landmarks[13].x * w, landmarks[13].y * h];
    const lower = [landmarks[14].x * w, landmarks[14].y * h];

    // Draw mouth landmark dots
    ctx.beginPath();
    ctx.arc(upper[0], upper[1], 3, 0, 2 * Math.PI);
    ctx.arc(lower[0], lower[1], 3, 0, 2 * Math.PI);
    ctx.fill();

    mar = distance(upper, lower);

    // Head Pose rotation angles
    const poseAngles = computeHeadPose(landmarks, w, h);
    x_angle = poseAngles.x_angle;
    y_angle = poseAngles.y_angle;

    // Phase transitions/sample collection
    if (calibrationState === CALIB_STATE_EYES) {
      eyeEarSamples.push(ear);
    } else if (calibrationState === CALIB_STATE_POSE) {
      poseXSamples.push(x_angle);
      poseYSamples.push(y_angle);
    }

    // Monitoring evaluations
    if (calibrationState === CALIB_STATE_DONE) {
      const adjustedX = x_angle - baselineXAngle;
      const adjustedY = y_angle - baselineYAngle;

      if (adjustedY < -10) pose = "Looking Left";
      else if (adjustedY > 10) pose = "Looking Right";
      else if (adjustedX < -10) pose = "Looking Down";
      else pose = "Forward";

      // Eye closing check
      if (ear < EYE_THRESHOLD) {
        if (closedStart === null) {
          closedStart = currentTime;
        }
        const closedDuration = currentTime - closedStart;
        if (closedDuration > 2.0) {
          drowsy = true;
        }
      } else {
        closedStart = null;
      }

      // Yawn check
      if (mar > 25.0) {
        yawning = true;
      }
    }
  }

  // Handle logging on state transitions
  if (faceDetectedThisFrame !== window.lastFaceDetectedState) {
    if (faceDetectedThisFrame) logEvent("Face detected.", "INFO");
    else logEvent("Face lost / not detected.", "WARNING");
    window.lastFaceDetectedState = faceDetectedThisFrame;
  }

  // Calibration state timer progression
  if (calibrationState !== CALIB_STATE_DONE) {
    if (faceDetectedThisFrame) {
      if (lastValidFrameTime !== null) {
        const delta = currentTime - lastValidFrameTime;
        calibElapsedTime += delta;
      }
      lastValidFrameTime = currentTime;

      if (calibrationState === CALIB_STATE_EYES) {
        if (calibElapsedTime >= CALIBRATION_DURATION) {
          if (eyeEarSamples.length > 0) {
            const avgEar = eyeEarSamples.reduce((a,b)=>a+b, 0) / eyeEarSamples.length;
            EYE_THRESHOLD = Math.min(Math.max(avgEar * 0.70, 0.15), 0.25);
            logEvent(`Open-eye calibration complete. Average EAR: ${avgEar.toFixed(4)}. Dynamic EYE_THRESHOLD set to ${EYE_THRESHOLD.toFixed(4)}`, "INFO");
          } else {
            EYE_THRESHOLD = 0.20;
            logEvent("No EAR samples collected. Using default EYE_THRESHOLD: 0.20", "WARNING");
          }
          calibrationState = CALIB_STATE_POSE;
          calibElapsedTime = 0.0;
          lastValidFrameTime = null;
        }
      } else if (calibrationState === CALIB_STATE_POSE) {
        if (calibElapsedTime >= CALIBRATION_DURATION) {
          if (poseXSamples.length > 0 && poseYSamples.length > 0) {
            baselineXAngle = poseXSamples.reduce((a,b)=>a+b, 0) / poseXSamples.length;
            baselineYAngle = poseYSamples.reduce((a,b)=>a+b, 0) / poseYSamples.length;
            logEvent(`Head-pose calibration complete. Baseline X: ${baselineXAngle.toFixed(2)}, Baseline Y: ${baselineYAngle.toFixed(2)}`, "INFO");
          } else {
            baselineXAngle = 0.0;
            baselineYAngle = 0.0;
            logEvent("No pose samples collected. Using default baseline (0.0, 0.0)", "WARNING");
          }
          calibrationState = CALIB_STATE_DONE;
          calibElapsedTime = 0.0;
          lastValidFrameTime = null;
        }
      }
    } else {
      lastValidFrameTime = null;
    }
  }

  // Distraction checks
  if (calibrationState === CALIB_STATE_DONE) {
    if (pose === "Looking Down") {
      if (distractedStartTime === null) {
        distractedStartTime = currentTime;
      }
      distractedDuration = currentTime - distractedStartTime;
      if (distractedDuration > 3.0) {
        distractedAlert = true;
      }
    } else {
      distractedStartTime = null;
    }

    if (phoneDuration > 2.0) {
      phoneUsageAlert = true;
    }

    // Status engine transitions
    if (drowsy) status = "DROWSY";
    else if (yawning) status = "FATIGUED";
    else if (phoneUsageAlert) status = "PHONE_USAGE";
    else if (distractedAlert) status = "DISTRACTED";
    else status = "SAFE";

    // Deduct safety score on transition
    if (status !== window.lastStatusState) {
      if (status === "SAFE") {
        logEvent(`Status changed: -> ${status}`, "INFO");
      } else {
        logEvent(`Status changed: -> ${status}`, "WARNING");
        if (status === "DROWSY") {
          driverScore = Math.max(0, driverScore - 30);
          logEvent(`Penalty applied: DROWSY (-30). Score: ${driverScore}`, "INFO");
        } else if (status === "PHONE_USAGE") {
          driverScore = Math.max(0, driverScore - 20);
          logEvent(`Penalty applied: PHONE_USAGE (-20). Score: ${driverScore}`, "INFO");
        } else if (status === "DISTRACTED") {
          driverScore = Math.max(0, driverScore - 15);
          logEvent(`Penalty applied: DISTRACTED (-15). Score: ${driverScore}`, "INFO");
        } else if (status === "FATIGUED") {
          driverScore = Math.max(0, driverScore - 10);
          logEvent(`Penalty applied: FATIGUED (-10). Score: ${driverScore}`, "INFO");
        }
      }
      window.lastStatusState = status;
    }

    // Safety recovery logic
    if (status === "SAFE") {
      if (safeStartTime === null) {
        safeStartTime = currentTime;
      } else if (currentTime - safeStartTime >= 5.0) {
        if (driverScore < 100) {
          driverScore = Math.min(100, driverScore + 1);
          logEvent(`Score recovered: +1. Score: ${driverScore}`, "INFO");
        }
        safeStartTime = currentTime;
      }
    } else {
      safeStartTime = null;
    }

    // Sound triggers for warnings
    if (["DROWSY", "PHONE_USAGE", "DISTRACTED"].includes(status)) {
      triggerAudioAlarm();
    }

    // Seatbelt logging checks
    if (seatbeltDetected !== window.lastSeatbeltState) {
      if (seatbeltDetected) logEvent("Seatbelt status changed: Worn", "INFO");
      else logEvent("Seatbelt status changed: Not Detected", "WARNING");
      window.lastSeatbeltState = seatbeltDetected;
    }
  } else {
    status = "CALIBRATION";
  }

  // ------------------------------------
  // Draw Canvas Overlays (HUD Dashboard)
  // ------------------------------------
  if (calibrationState !== CALIB_STATE_DONE) {
    // 1. Translucent Bottom Card
    ctx.fillStyle = 'rgba(28, 25, 23, 0.75)';
    const cardX1 = 40;
    const cardY1 = h - 170;
    const cardW = w - 80;
    const cardH = 140;
    
    // Draw card
    ctx.fillRect(cardX1, cardY1, cardW, cardH);
    
    // Glowing neon blue/cyan border
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.strokeRect(cardX1, cardY1, cardW, cardH);

    // Header Text
    ctx.fillStyle = '#c8c8c8';
    ctx.font = '14px sans-serif';
    ctx.fillText("DAMTS SYSTEM CALIBRATION", cardX1 + 20, cardY1 + 30);

    let phaseTitle = "";
    let phaseInstruction = "";
    let phaseMetric = "";

    if (calibrationState === CALIB_STATE_EYES) {
      phaseTitle = "PHASE 1: EYE CALIBRATION";
      phaseInstruction = "Look straight ahead and keep your eyes open";
      phaseMetric = `Current EAR: ${ear.toFixed(2)}`;
    } else {
      phaseTitle = "PHASE 2: HEAD POSE CALIBRATION";
      phaseInstruction = "Maintain normal driving posture, look straight ahead";
      phaseMetric = faceDetectedThisFrame ? `Raw Pitch: ${x_angle.toFixed(1)}, Yaw: ${y_angle.toFixed(1)}` : "Raw angles: N/A";
    }

    // Title & Instructions
    ctx.fillStyle = '#06b6d4';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(phaseTitle, cardX1 + 20, cardY1 + 60);

    ctx.fillStyle = '#ffffff';
    ctx.font = '13px sans-serif';
    ctx.fillText(phaseInstruction, cardX1 + 20, cardY1 + 85);

    // Timer and face detection validation
    const remainingTime = Math.max(0.0, CALIBRATION_DURATION - calibElapsedTime);
    let barColor = '#10b981'; // Green
    
    if (faceDetectedThisFrame) {
      ctx.fillStyle = '#10b981';
      ctx.fillText(`Status: Calibrating (${remainingTime.toFixed(1)}s left) | ${phaseMetric}`, cardX1 + 20, cardY1 + 110);
    } else {
      ctx.fillStyle = '#ef4444';
      ctx.fillText("Status: NO FACE DETECTED (Calibration Paused)", cardX1 + 20, cardY1 + 110);
      barColor = '#f59e0b'; // Orange when paused
    }

    // Draw calibration progress bar
    const barX = cardX1 + 20;
    const barY = cardY1 + 120;
    const barW = cardW - 40;
    const barH = 8;
    
    ctx.fillStyle = '#323232';
    ctx.fillRect(barX, barY, barW, barH);
    
    const progress = Math.min(1.0, calibElapsedTime / CALIBRATION_DURATION);
    ctx.fillStyle = barColor;
    ctx.fillRect(barX, barY, barW * progress, barH);
  } else {
    // 2. Monitoring HUD Dashboard overlay (Top Left)
    let dashboardColor = '#10b981'; // Green
    if (["DROWSY", "PHONE_USAGE"].includes(status)) {
      dashboardColor = '#ef4444'; // Red
    } else if (["FATIGUED", "DISTRACTED"].includes(status)) {
      dashboardColor = '#f59e0b'; // Yellow/Orange
    }

    const phoneStr = phoneDuration > 0.0 ? 
      `${phoneDetected ? 'Yes' : 'No'} (${phoneDuration.toFixed(1)}s)` : "No";

    const dashboardLines = [
      "===================",
      `STATUS : ${status}`,
      `EAR    : ${ear.toFixed(2)}`,
      `MAR    : ${mar.toFixed(0)}`,
      `POSE   : ${pose}`,
      `PHONE  : ${phoneStr}`,
      `SCORE  : ${driverScore}`,
      "==================="
    ];

    ctx.fillStyle = dashboardColor;
    ctx.font = 'bold 15px monospace';
    let yOffset = 40;
    dashboardLines.forEach(line => {
      ctx.fillText(line, 20, yOffset);
      yOffset += 24;
    });

    // Seatbelt indicators
    const sbText = seatbeltDetected ? "SEATBELT: Worn" : "SEATBELT: Not Detected";
    const sbColor = seatbeltDetected ? '#10b981' : '#ef4444';
    ctx.fillStyle = sbColor;
    ctx.fillText(sbText, 20, yOffset + 10);

    // Warning durations
    let warningY = yOffset + 40;
    if (ear < EYE_THRESHOLD && closedStart !== null) {
      const closedDur = currentTime - closedStart;
      ctx.fillStyle = '#f59e0b';
      ctx.fillText(`Closed: ${closedDur.toFixed(1)}s`, 20, warningY);
      warningY += 24;
    }

    if (pose === "Looking Down" && distractedStartTime !== null) {
      ctx.fillStyle = '#f59e0b';
      ctx.fillText(`Distracted: ${distractedDuration.toFixed(1)}s`, 20, warningY);
    }
  }

  // ------------------------------------
  // Sync Data to Web DOM Dashboard Panels
  // ------------------------------------
  hudScore.textContent = driverScore;
  hudScoreBar.style.width = `${driverScore}%`;
  
  if (driverScore > 70) hudScoreBar.style.backgroundColor = 'var(--accent-green)';
  else if (driverScore > 40) hudScoreBar.style.backgroundColor = 'var(--accent-yellow)';
  else hudScoreBar.style.backgroundColor = 'var(--accent-red)';
  
  hudState.textContent = status;
  hudState.className = "telemetry-value " + 
    (status === "SAFE" ? "value-safe" : 
     (status === "CALIBRATION" ? "value-normal" : 
      (["DROWSY", "PHONE_USAGE"].includes(status) ? "value-danger" : "value-warning")));

  hudEar.textContent = ear.toFixed(2);
  hudEar.className = "telemetry-value " + 
    (calibrationState === CALIB_STATE_DONE && ear < EYE_THRESHOLD ? "value-warning" : "value-normal");

  hudMar.textContent = mar.toFixed(0);
  hudMar.className = "telemetry-value " + (mar > 25.0 ? "value-warning" : "value-normal");

  hudPose.textContent = pose;
  hudPose.className = "telemetry-value " + (pose !== "Forward" ? "value-warning" : "value-normal");

  hudPhone.textContent = phoneDetected ? "Yes" : "No";
  hudPhone.className = "telemetry-value " + (phoneDetected ? "value-danger" : "value-normal");

  hudSeatbelt.textContent = seatbeltDetected ? "Worn" : "Not Detected";
  hudSeatbelt.className = "telemetry-value " + (seatbeltDetected ? "value-safe" : "value-danger");

  // Sync header state border colors
  const videoCard = document.getElementById('video-feed-card');
  const headerBadge = document.getElementById('system-header-badge');
  const badgeDot = document.getElementById('system-badge-dot');
  const badgeText = document.getElementById('system-badge-text');

  if (["DROWSY", "PHONE_USAGE", "DISTRACTED"].includes(status)) {
    videoCard.classList.add('alert-active');
    headerBadge.style.color = 'var(--accent-red)';
    headerBadge.style.background = 'rgba(239, 68, 68, 0.1)';
    headerBadge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
    badgeDot.style.backgroundColor = 'var(--accent-red)';
    badgeDot.style.boxShadow = '0 0 8px var(--accent-red)';
    badgeText.textContent = `ALERT: ${status}`;
  } else {
    videoCard.classList.remove('alert-active');
    headerBadge.style.color = status === "CALIBRATION" ? 'var(--accent-yellow)' : 'var(--accent-green)';
    headerBadge.style.background = status === "CALIBRATION" ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)';
    headerBadge.style.borderColor = status === "CALIBRATION" ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.2)';
    badgeDot.style.backgroundColor = status === "CALIBRATION" ? 'var(--accent-yellow)' : 'var(--accent-green)';
    badgeDot.style.boxShadow = status === "CALIBRATION" ? '0 0 8px var(--accent-yellow)' : '0 0 8px var(--accent-green)';
    badgeText.textContent = status === "CALIBRATION" ? "CALIBRATING" : "MONITORING ACTIVE";
  }

  // Continue Processing Frame loop
  if (isRunning) {
    animationFrameId = requestAnimationFrame(captureLoop);
  }
}

// ------------------------------------
// Start / Pause / Stop Session Controls
// ------------------------------------

async function startSession() {
  if (isRunning) return;

  try {
    logEvent("Requesting webcam device stream...", "INFO");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        width: { ideal: 640 }, 
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    });
    
    video.srcObject = stream;
    // Wait for stream metadata to load
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        resolve();
      };
    });
    
    video.play();
    
    // Set Canvas size matching the actual video feed dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Reset States
    isRunning = true;
    calibrationState = CALIB_STATE_EYES;
    calibElapsedTime = 0.0;
    eyeEarSamples = [];
    poseXSamples = [];
    poseYSamples = [];
    closedStart = null;
    phoneAccumulatedTime = 0.0;
    distractedStartTime = null;
    driverScore = 100;
    
    window.lastStatusState = null;
    window.lastSeatbeltState = null;
    window.lastFaceDetectedState = null;
    lastFrameTime = Date.now() / 1000;

    // Start UI stream recorder
    recordedChunks = [];
    const canvasStream = canvas.captureStream(20); // 20 FPS
    mediaRecorder = new MediaRecorder(canvasStream, { mimeType: 'video/webm;codecs=vp9' });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = exportVideo;
    mediaRecorder.start();

    // Toggle DOM elements
    btnStart.setAttribute('disabled', 'true');
    btnPause.removeAttribute('disabled');
    btnStop.removeAttribute('disabled');
    recIndicator.style.display = 'flex';
    document.getElementById('setup-guide').style.display = 'none';

    logEvent("DMS System started. Calibration phase 1 (EYES) initialized.", "INFO");
    
    // Fire Loop
    animationFrameId = requestAnimationFrame(captureLoop);
  } catch (err) {
    logEvent(`Webcam Access Denied: ${err.message}`, "ERROR");
    alert("Could not access camera. Please grant camera permissions and retry.");
  }
}

function pauseSession() {
  if (!isRunning) return;
  isRunning = false;
  cancelAnimationFrame(animationFrameId);
  video.pause();
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.pause();
  }

  btnStart.removeAttribute('disabled');
  btnPause.setAttribute('disabled', 'true');
  logEvent("Session paused by user.", "INFO");
}

function stopSession() {
  if (!isRunning && !mediaRecorder) return;
  
  isRunning = false;
  cancelAnimationFrame(animationFrameId);
  video.pause();
  
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
  }

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }

  btnStart.removeAttribute('disabled');
  btnPause.setAttribute('disabled', 'true');
  btnStop.setAttribute('disabled', 'true');
  btnDownloadLogs.removeAttribute('disabled');
  btnDownloadTelemetry.removeAttribute('disabled');
  recIndicator.style.display = 'none';
  
  // Reset HUD badge
  const headerBadge = document.getElementById('system-header-badge');
  const badgeDot = document.getElementById('system-badge-dot');
  const badgeText = document.getElementById('system-badge-text');
  headerBadge.style.color = 'var(--text-secondary)';
  headerBadge.style.background = 'rgba(255,255,255,0.05)';
  headerBadge.style.borderColor = 'var(--panel-border)';
  badgeDot.style.backgroundColor = 'var(--text-secondary)';
  badgeDot.style.boxShadow = 'none';
  badgeText.textContent = "SYSTEM STANDBY";

  logEvent("DMS Session ended. Exited stream recorder. Export files compiled.", "INFO");
}

function exportVideo() {
  if (recordedChunks.length === 0) return;
  const blob = new Blob(recordedChunks, { type: 'video/mp4' });
  const url = URL.createObjectURL(blob);
  
  // Auto-download video session
  const a = document.createElement('a');
  a.href = url;
  a.download = 'demo.mp4';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  logEvent("Session recording exported to demo.mp4", "INFO");
}

function triggerLogsDownload() {
  const blob = new Blob([systemLogs.join('\n')], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = 'dms_events.log';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  logEvent("Log file exported successfully.", "INFO");
}

function triggerTelemetryDownload() {
  let csv = 'timestamp,confidence,box_width,box_height,is_confirmed\n';
  telemetryData.forEach(row => {
    csv += `${row.timestamp},${row.confidence},${row.width},${row.height},${row.confirmed}\n`;
  });
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = 'phone_confidence_telemetry.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  logEvent("Telemetry history exported successfully.", "INFO");
}

// ------------------------------------
// Event Listeners Configuration
// ------------------------------------

btnStart.addEventListener('click', startSession);
btnPause.addEventListener('click', pauseSession);
btnStop.addEventListener('click', stopSession);

btnMute.addEventListener('click', () => {
  isMuted = !isMuted;
  const muteIcon = document.getElementById('mute-icon');
  const muteText = document.getElementById('mute-text');
  
  if (isMuted) {
    btnMute.className = "btn-secondary btn-danger";
    muteIcon.setAttribute('data-lucide', 'volume-x');
    muteText.textContent = "Alerts: OFF";
    logEvent("Audio warnings muted.", "WARNING");
  } else {
    btnMute.className = "btn-secondary";
    muteIcon.setAttribute('data-lucide', 'volume-2');
    muteText.textContent = "Alerts: ON";
    logEvent("Audio warnings unmuted.", "INFO");
  }
  lucide.createIcons();
});

btnDownloadLogs.addEventListener('click', triggerLogsDownload);
btnDownloadTelemetry.addEventListener('click', triggerTelemetryDownload);

// On Start
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  initializeApp();
});
