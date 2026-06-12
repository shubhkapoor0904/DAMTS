import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import threading
import sys
import os
import subprocess
from ultralytics import YOLO

# Conditionally import winsound on Windows to prevent startup crash on Linux/macOS
if sys.platform == "win32":
    import winsound

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dms_events.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DMS")

# Alarm system state
alarm_playing = False

def play_alarm_sound():
    global alarm_playing
    try:
        if sys.platform == "win32":
            # Frequency 2000 Hz, duration 400 ms
            winsound.Beep(2000, 400)
        elif sys.platform == "darwin":
            # macOS system beep
            subprocess.run(["osascript", "-e", "beep"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Linux and other Unix-like systems
            sound_played = False
            
            # Common system alert sound paths on Linux
            common_sounds = [
                "/usr/share/sounds/freedesktop/stereo/bell.oga",
                "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga",
                "/usr/share/sounds/freedesktop/stereo/complete.oga",
                "/usr/share/sounds/freedesktop/stereo/dialog-warning.oga",
                "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"
            ]
            
            # Find the first existing sound file
            sound_file = next((path for path in common_sounds if os.path.exists(path)), None)
            
            if sound_file:
                # Try playing with paplay (PulseAudio), pw-play (PipeWire), or aplay (ALSA)
                for player in ["paplay", "pw-play", "aplay"]:
                    try:
                        res = subprocess.run([player, sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if res.returncode == 0:
                            sound_played = True
                            break
                    except Exception:
                        continue
            
            # Fallback 1: Try running standard 'beep' command
            if not sound_played:
                try:
                    res = subprocess.run(["beep", "-f", "2000", "-d", "400"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        sound_played = True
                except Exception:
                    pass

            # Fallback 2: Try text-to-speech 'spd-say'
            if not sound_played:
                try:
                    res = subprocess.run(["spd-say", "-r", "50", "warning"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        sound_played = True
                except Exception:
                    pass
            
            # Fallback 3: Send ASCII Bell character to stdout
            if not sound_played:
                sys.stdout.write('\a')
                sys.stdout.flush()
                
    except Exception as e:
        logger.error(f"Failed to play alert sound: {e}")
    finally:
        alarm_playing = False

def trigger_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        t = threading.Thread(target=play_alarm_sound, daemon=True)
        t.start()

# ------------------
# YOLO
# ------------------
model = YOLO("best.pt")

# ------------------
# MediaPipe
# ------------------
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
UPPER_LIP = 13
LOWER_LIP = 14
EYE_THRESHOLD = 0.20

# Calibration Configuration and States
CALIB_STATE_EYES = "EYES"
CALIB_STATE_POSE = "POSE"
CALIB_STATE_DONE = "DONE"

calibration_state = CALIB_STATE_EYES
CALIBRATION_DURATION = 3.0  # Duration in seconds for each phase

# Accumulators for samples
eye_ear_samples = []
pose_x_samples = []
pose_y_samples = []

# Time tracking for valid calibration frames
calib_elapsed_time = 0.0
last_valid_frame_time = None

# Baseline values (computed during calibration)
baseline_x_angle = 0.0
baseline_y_angle = 0.0

closed_start = None
phone_start_time = None
last_phone_detection_time = None
distracted_start_time = None
driver_score = 100
safe_start_time = None

# YOLO Confidence and Tracking Parameters
YOLO_CONF_BASE = 0.20      # Minimum confidence for YOLO inference
YOLO_CONF_PHONE = 0.35     # Threshold for phone detection to avoid false positives
YOLO_CONF_SEATBELT = 0.25  # Threshold for seatbelt detection
PHONE_GRACE_PERIOD = 0.8   # Time in seconds to tolerate transient detection misses


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


cap = cv2.VideoCapture(0)

last_status = None
last_seatbelt_detected = None
last_face_detected = None
out = None

logger.info("DMS System initialized. Starting video capture...")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        h, w, _ = frame.shape

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 100:
                fps = 20.0
            out = cv2.VideoWriter('demo.mp4', fourcc, fps, (w, h))
            logger.info(f"Video recorder initialized (demo.mp4). Output resolution: {w}x{h} @ {fps} FPS")

        status = "SAFE"
        pose = "Forward"
        drowsy = False
        yawning = False
        seatbelt_detected = False
        phone_usage_alert = False
        distracted_alert = False
        ear = 0.0
        mar = 0.0
        x_angle = 0.0
        y_angle = 0.0
        phone_duration = 0.0
        distracted_duration = 0.0

        h, w, _ = frame.shape

        # ------------------
        # YOLO (Only run if calibration is complete)
        # ------------------
        phone_detected = False
        seatbelt_detected = False

        if calibration_state == CALIB_STATE_DONE:
            results = model(frame, conf=YOLO_CONF_BASE, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])

                    if label == "Dist_mob" and conf >= YOLO_CONF_PHONE:
                        phone_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"Phone: {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                    elif label == "Set_belt" and conf >= YOLO_CONF_SEATBELT:
                        seatbelt_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Seatbelt: {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

            # ------------------
            # Phone Detection Duration Tracking with Temporal Smoothing
            # ------------------
            current_time = time.time()
            if phone_detected:
                last_phone_detection_time = current_time
                if phone_start_time is None:
                    phone_start_time = current_time
                phone_duration = current_time - phone_start_time
                if phone_duration > 2.0:
                    phone_usage_alert = True
            else:
                # To handle transient drops (e.g. motion blur, hand occluding camera momentarily),
                # allow a grace period where we preserve the start time.
                if last_phone_detection_time is not None and (current_time - last_phone_detection_time) <= PHONE_GRACE_PERIOD:
                    phone_duration = current_time - phone_start_time
                    if phone_duration > 2.0:
                        phone_usage_alert = True
                else:
                    phone_start_time = None
                    last_phone_detection_time = None

        # ------------------
        # FaceMesh
        # ------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb)

        face_detected_this_frame = False

        if mesh_results.multi_face_landmarks:
            face_detected_this_frame = True
            face_landmarks = mesh_results.multi_face_landmarks[0]

            # ------------------
            # Head Pose Estimation
            # ------------------
            face_2d = []
            face_3d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d,
                face_2d,
                cam_matrix,
                dist_matrix
            )

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, *_ = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360

            # During pose calibration, accumulate sample angles
            if calibration_state == CALIB_STATE_POSE:
                pose_x_samples.append(x_angle)
                pose_y_samples.append(y_angle)

            # Determine pose relative to baseline if calibration is done
            if calibration_state == CALIB_STATE_DONE:
                adjusted_x = x_angle - baseline_x_angle
                adjusted_y = y_angle - baseline_y_angle

                if adjusted_y < -10:
                    pose = "Looking Left"
                elif adjusted_y > 10:
                    pose = "Looking Right"
                elif adjusted_x < -10:
                    pose = "Looking Down"
            else:
                pose = "Forward"

            # ------------------
            # EAR Drowsiness Calculations
            # ------------------
            eye_points = []
            for idx in LEFT_EYE:
                ex = int(face_landmarks.landmark[idx].x * w)
                ey = int(face_landmarks.landmark[idx].y * h)
                eye_points.append((ex, ey))
                cv2.circle(frame, (ex, ey), 2, (0, 255, 0), -1)

            vertical1 = distance(eye_points[1], eye_points[5])
            vertical2 = distance(eye_points[2], eye_points[4])
            horizontal = distance(eye_points[0], eye_points[3])
            ear = (vertical1 + vertical2) / (2.0 * horizontal)

            # During eye calibration, accumulate EAR samples
            if calibration_state == CALIB_STATE_EYES:
                eye_ear_samples.append(ear)

            # Standard monitoring checks
            if calibration_state == CALIB_STATE_DONE:
                if ear < EYE_THRESHOLD:
                    if closed_start is None:
                        closed_start = time.time()
                    closed_duration = time.time() - closed_start
                    if closed_duration > 2:
                        drowsy = True
                else:
                    closed_start = None

            # ------------------
            # Yawn Detection
            # ------------------
            upper = face_landmarks.landmark[UPPER_LIP]
            lower = face_landmarks.landmark[LOWER_LIP]

            p1 = (int(upper.x * w), int(upper.y * h))
            p2 = (int(lower.x * w), int(lower.y * h))

            cv2.circle(frame, p1, 3, (0, 255, 0), -1)
            cv2.circle(frame, p2, 3, (0, 255, 0), -1)

            mar = distance(p1, p2)

            if calibration_state == CALIB_STATE_DONE:
                if mar > 25:
                    yawning = True

        # Log face detection state changes
        if face_detected_this_frame != last_face_detected:
            if face_detected_this_frame:
                logger.info("Face detected.")
            else:
                logger.warning("Face lost / not detected.")
            last_face_detected = face_detected_this_frame

        # Update calibration timers and transition logic
        if calibration_state != CALIB_STATE_DONE:
            if face_detected_this_frame:
                current_time = time.time()
                if last_valid_frame_time is not None:
                    delta = current_time - last_valid_frame_time
                    calib_elapsed_time += delta
                last_valid_frame_time = current_time

                if calibration_state == CALIB_STATE_EYES:
                    if calib_elapsed_time >= CALIBRATION_DURATION:
                        if len(eye_ear_samples) > 0:
                            avg_ear = sum(eye_ear_samples) / len(eye_ear_samples)
                            EYE_THRESHOLD = np.clip(avg_ear * 0.70, 0.15, 0.25)
                            logger.info(f"Open-eye calibration complete. Average EAR: {avg_ear:.4f}. Dynamic EYE_THRESHOLD set to {EYE_THRESHOLD:.4f}")
                        else:
                            EYE_THRESHOLD = 0.20
                            logger.warning("No EAR samples collected. Using default EYE_THRESHOLD: 0.20")
                        
                        calibration_state = CALIB_STATE_POSE
                        calib_elapsed_time = 0.0
                        last_valid_frame_time = None

                elif calibration_state == CALIB_STATE_POSE:
                    if calib_elapsed_time >= CALIBRATION_DURATION:
                        if len(pose_x_samples) > 0 and len(pose_y_samples) > 0:
                            baseline_x_angle = sum(pose_x_samples) / len(pose_x_samples)
                            baseline_y_angle = sum(pose_y_samples) / len(pose_y_samples)
                            logger.info(f"Head-pose calibration complete. Baseline X: {baseline_x_angle:.2f}, Baseline Y: {baseline_y_angle:.2f}")
                        else:
                            baseline_x_angle = 0.0
                            baseline_y_angle = 0.0
                            logger.warning("No pose samples collected. Using default baseline (0.0, 0.0)")
                        
                        calibration_state = CALIB_STATE_DONE
                        calib_elapsed_time = 0.0
                        last_valid_frame_time = None
            else:
                last_valid_frame_time = None

        # ------------------
        # Distraction Duration Tracking
        # ------------------
        if calibration_state == CALIB_STATE_DONE:
            if pose == "Looking Down":
                if distracted_start_time is None:
                    distracted_start_time = time.time()
                distracted_duration = time.time() - distracted_start_time
                if distracted_duration > 3:
                    distracted_alert = True
            else:
                distracted_start_time = None

            # ------------------
            # Decision Engine
            # ------------------
            if drowsy:
                status = "DROWSY"
            elif yawning:
                status = "FATIGUED"
            elif phone_usage_alert:
                status = "PHONE_USAGE"
            elif distracted_alert:
                status = "DISTRACTED"
            else:
                status = "SAFE"

            # Log status change and update driver score
            if status != last_status:
                if status == "SAFE":
                    logger.info(f"Status changed: -> {status}")
                else:
                    logger.warning(f"Status changed: -> {status}")
                    # Deduct penalty once on transition
                    if status == "DROWSY":
                        driver_score = max(0, driver_score - 30)
                        logger.info(f"Penalty applied: DROWSY (-30). Score: {driver_score}")
                    elif status == "PHONE_USAGE":
                        driver_score = max(0, driver_score - 20)
                        logger.info(f"Penalty applied: PHONE_USAGE (-20). Score: {driver_score}")
                    elif status == "DISTRACTED":
                        driver_score = max(0, driver_score - 15)
                        logger.info(f"Penalty applied: DISTRACTED (-15). Score: {driver_score}")
                    elif status == "FATIGUED":
                        driver_score = max(0, driver_score - 10)
                        logger.info(f"Penalty applied: FATIGUED (-10). Score: {driver_score}")
                last_status = status

            # Recovery logic: if status is SAFE, slowly recover score (+1 point every 5 seconds)
            if status == "SAFE":
                if safe_start_time is None:
                    safe_start_time = time.time()
                elif time.time() - safe_start_time >= 5.0:
                    if driver_score < 100:
                        driver_score = min(100, driver_score + 1)
                        logger.info(f"Score recovered: +1. Score: {driver_score}")
                    safe_start_time = time.time()
            else:
                safe_start_time = None

            # Trigger audio alarm for critical states (DROWSY, PHONE_USAGE, DISTRACTED)
            if status in ["DROWSY", "PHONE_USAGE", "DISTRACTED"]:
                trigger_alarm()

            # Log seatbelt status change
            if seatbelt_detected != last_seatbelt_detected:
                if seatbelt_detected:
                    logger.info("Seatbelt status changed: Worn")
                else:
                    logger.warning("Seatbelt status changed: Not Detected")
                last_seatbelt_detected = seatbelt_detected
        else:
            status = "CALIBRATION"

        # ------------------
        # Display (Dashboard / Calibration Overlay)
        # ------------------
        if calibration_state != CALIB_STATE_DONE:
            # Draw premium glassmorphism calibration card at the bottom
            overlay = frame.copy()
            
            # Bottom card coordinates
            card_x1, card_y1 = 40, h - 170
            card_x2, card_y2 = w - 40, h - 30
            
            # Fill with dark slate (BGR: (28, 25, 23))
            cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (28, 25, 23), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Glowing neon blue/cyan border (BGR: (255, 200, 0))
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (255, 200, 0), 2)
            
            # Calibration Header
            cv2.putText(
                frame,
                "DAMTS SYSTEM CALIBRATION",
                (card_x1 + 20, card_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                lineType=cv2.LINE_AA
            )
            
            if calibration_state == CALIB_STATE_EYES:
                title = "PHASE 1: EYE CALIBRATION"
                instruction = "Look straight ahead and keep your eyes open"
                metric_text = f"Current EAR: {ear:.2f}"
            else:
                title = "PHASE 2: HEAD POSE CALIBRATION"
                instruction = "Maintain normal driving posture, look straight ahead"
                metric_text = f"Raw Pitch: {x_angle:.1f}, Yaw: {y_angle:.1f}" if face_detected_this_frame else "Raw angles: N/A"
            
            cv2.putText(
                frame,
                title,
                (card_x1 + 20, card_y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255), # Neon Yellow/Cyan
                2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame,
                instruction,
                (card_x1 + 20, card_y1 + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )
            
            remaining_time = max(0.0, CALIBRATION_DURATION - calib_elapsed_time)
            
            if face_detected_this_frame:
                status_text = f"Status: Calibrating ({remaining_time:.1f}s left) | {metric_text}"
                status_color = (0, 255, 100) # Neon Green
            else:
                status_text = "Status: NO FACE DETECTED (Calibration Paused)"
                status_color = (0, 0, 255) # Red
                
            cv2.putText(
                frame,
                status_text,
                (card_x1 + 20, card_y1 + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
                lineType=cv2.LINE_AA
            )
            
            # Progress Bar
            bar_x1 = card_x1 + 20
            bar_y1 = card_y2 - 25
            bar_x2 = card_x2 - 20
            bar_y2 = card_y2 - 15
            
            # Progress bar background
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
            
            progress_ratio = min(1.0, calib_elapsed_time / CALIBRATION_DURATION)
            fill_x2 = int(bar_x1 + (bar_x2 - bar_x1) * progress_ratio)
            fill_color = (0, 255, 100) if face_detected_this_frame else (0, 140, 255) # Green / Orange
            
            if fill_x2 > bar_x1:
                cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), fill_color, -1)
        else:
            # Draw standard dashboard
            if phone_detected:
                phone_str = f"Yes ({phone_duration:.1f}s)"
            else:
                phone_str = "No"

            # Determine dashboard color based on status
            if status in ["DROWSY", "PHONE_USAGE"]:
                dash_color = (0, 0, 255)  # Red
            elif status in ["FATIGUED", "DISTRACTED"]:
                dash_color = (0, 255, 255)  # Yellow/Orange
            else:
                dash_color = (0, 255, 0)  # Green

            dashboard_lines = [
                "===================",
                f"STATUS : {status}",
                f"EAR    : {ear:.2f}",
                f"MAR    : {mar:.0f}",
                f"POSE   : {pose}",
                f"PHONE  : {phone_str}",
                f"SCORE  : {driver_score}",
                "==================="
            ]

            y_offset = 40
            for line in dashboard_lines:
                cv2.putText(
                    frame,
                    line,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    dash_color,
                    2
                )
                y_offset += 30

            # Draw seatbelt status just below the dashboard
            sb_text = "SEATBELT: Worn" if seatbelt_detected else "SEATBELT: Not Detected"
            sb_color = (0, 255, 0) if seatbelt_detected else (0, 0, 255)
            cv2.putText(
                frame,
                sb_text,
                (20, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                sb_color,
                2
            )

            # Draw eyes closed and distracted duration alerts
            current_y = y_offset + 40
            if ear < EYE_THRESHOLD and closed_start is not None:
                closed_dur = time.time() - closed_start
                cv2.putText(
                    frame,
                    f"Closed: {closed_dur:.1f}s",
                    (20, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                current_y += 30

            if pose == "Looking Down" and distracted_start_time is not None:
                cv2.putText(
                    frame,
                    f"Distracted: {distracted_duration:.1f}s",
                    (20, current_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                current_y += 30

        cv2.imshow("DAMTS", frame)

        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            logger.info("ESC key pressed. Exiting...")
            break

logger.info("DMS System shutting down. Releasing camera and closing windows...")
cap.release()
if out is not None:
    out.release()
    logger.info("Video recording saved to demo.mp4")
cv2.destroyAllWindows()
logger.info("DMS System shutdown complete.")