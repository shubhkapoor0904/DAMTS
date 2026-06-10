import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

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

closed_start = None


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


cap = cv2.VideoCapture(0)

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

        status = "SAFE"
        pose = "Forward"
        drowsy = False
        yawning = False
        seatbelt_detected = False

        h, w, _ = frame.shape

        # ------------------
        # YOLO
        # ------------------
        results = model(frame, conf=0.15, verbose=False)

        phone_detected = False
        seatbelt_detected = False

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                if label == "Dist_mob":
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
                elif label == "Set_belt":
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
        # FaceMesh
        # ------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:

            face_landmarks = mesh_results.multi_face_landmarks[0]

            # ------------------
            # Head Pose
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

            if y_angle < -10:
                pose = "Looking Left"

            elif y_angle > 10:
                pose = "Looking Right"

            elif x_angle < -10:
                pose = "Looking Down"

            # ------------------
            # EAR Drowsiness
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

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            THRESHOLD = 0.20

            if ear < THRESHOLD:

                if closed_start is None:
                    closed_start = time.time()

                closed_duration = time.time() - closed_start

                cv2.putText(
                    frame,
                    f"Closed: {closed_duration:.1f}s",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                    )

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

            cv2.putText(
                frame,
                f"MAR: {mar:.0f}",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if mar > 25:
                yawning = True

        # ------------------
        # Decision Engine
        # ------------------
        if drowsy:
            status = "DROWSY"

        elif yawning:
            status = "FATIGUED"

        elif phone_detected:
            status = "PHONE_USAGE"

        elif pose == "Looking Down":
            status = "DISTRACTED"

        else:
            status = "SAFE"

        # ------------------
        # Display
        # ------------------
        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"POSE: {pose}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        sb_text = "SEATBELT: Worn" if seatbelt_detected else "SEATBELT: Not Detected"
        sb_color = (0, 255, 0) if seatbelt_detected else (0, 0, 255)
        cv2.putText(
            frame,
            sb_text,
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            sb_color,
            2
        )

        cv2.imshow("DAMTS", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()