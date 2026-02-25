import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

base_options = python.BaseOptions(
    model_asset_path='pose_landmarker_lite.task'
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(options)

labels_dict = {
    0: 'Berdiri',
    1: 'Duduk'

}


POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (23,24),
    (11,23),(12,24),
    (23,25),(25,27),
    (24,26),(26,28)
]


cap = cv2.VideoCapture(0)

print("Mulai deteksi pose tubuh (tekan Q untuk keluar)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    result = detector.detect(mp_image)

    if result.pose_landmarks:

        pose_landmarks = result.pose_landmarks[0]

        x_ = []
        y_ = []
        data_aux = []


        for lm in pose_landmarks:
            x_.append(lm.x)
            y_.append(lm.y)


        for lm in pose_landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


        for start, end in POSE_CONNECTIONS:
            x1 = int(pose_landmarks[start].x * w)
            y1 = int(pose_landmarks[start].y * h)
            x2 = int(pose_landmarks[end].x * w)
            y2 = int(pose_landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


        for lm in pose_landmarks:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))


        if len(data_aux) == 66:  # 33 landmark x 2
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = labels_dict.get(int(prediction[0]), 'Unknown')

            cv2.putText(
                frame,
                predicted_label,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

    cv2.imshow("Pose Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()