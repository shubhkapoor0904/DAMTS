from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np

# ------------------
# YOLO
# ------------------
model = YOLO("best.pt")

# ------------------
# MediaPipe
# ------------------
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        status = "SAFE"

        # ------------------
        # YOLO
        # ------------------
        results = model(frame, verbose=False)

        phone_detected = False

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])

                label = model.names[cls]

                if label == "Dist_mob":
                    phone_detected = True

        # ------------------
        # Head Pose
        # ------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose = "Forward"

        mesh_results = face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:

            face_landmarks = mesh_results.multi_face_landmarks[0]

            face_2d = []
            face_3d = []

            h, w, _ = frame.shape

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
        # Decision Engine
        # ------------------

        if phone_detected:
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
            status,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("DAMTS", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()