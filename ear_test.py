import cv2
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

cap = cv2.VideoCapture(0)

closed_start = None

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            points = []

            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR Calculation
            vertical1 = distance(points[1], points[5])
            vertical2 = distance(points[2], points[4])

            horizontal = distance(points[0], points[3])

            ear = (vertical1 + vertical2) / (2.0 * horizontal)

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Drowsiness Detection
            THRESHOLD = 0.20

            if ear < THRESHOLD:

                if closed_start is None:
                    closed_start = time.time()

                closed_duration = time.time() - closed_start

                cv2.putText(
                    frame,
                    f"Closed: {closed_duration:.1f}s",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                if closed_duration > 2:

                    cv2.putText(
                        frame,
                        "DROWSY ALERT!",
                        (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

            else:
                closed_start = None

        cv2.imshow("DAMTS Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()