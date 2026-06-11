import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

UPPER_LIP = 13
LOWER_LIP = 14

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

cap = cv2.VideoCapture(0)

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

            face = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            upper = face.landmark[UPPER_LIP]
            lower = face.landmark[LOWER_LIP]

            p1 = (int(upper.x*w), int(upper.y*h))
            p2 = (int(lower.x*w), int(lower.y*h))

            cv2.circle(frame, p1, 3, (0,255,0), -1)
            cv2.circle(frame, p2, 3, (0,255,0), -1)

            mar = distance(p1, p2)

            cv2.putText(
                frame,
                f"MAR: {mar:.0f}",
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            if mar > 25:

                cv2.putText(
                    frame,
                    "YAWNING",
                    (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

        cv2.imshow("Yawn Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()