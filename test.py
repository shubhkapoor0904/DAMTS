import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

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

            print("Face Detected")

            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = frame.shape

                nose = face_landmarks.landmark[1]

                x = int(nose.x * w)
                y = int(nose.y * h)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("DAMTS FaceMesh Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()