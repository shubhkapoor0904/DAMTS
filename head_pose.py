import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            break

        image = cv2.flip(image, 1)

        img_h, img_w, img_c = image.shape

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            face_2d = []
            face_3d = []

            for idx, lm in enumerate(face_landmarks.landmark):

                if idx in [33, 263, 1, 61, 291, 199]:

                    x = int(lm.x * img_w)
                    y = int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = img_w

            cam_matrix = np.array(
                [
                    [focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]
                ]
            )

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d,
                face_2d,
                cam_matrix,
                dist_matrix
            )

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, *_ = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            text = "Forward"

            if y < -10:
                text = "Looking Left"

            elif y > 10:
                text = "Looking Right"

            elif x < -10:
                text = "Looking Down"

            elif x > 10:
                text = "Looking Up"

            cv2.putText(
                image,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Head Pose", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()