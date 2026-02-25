import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===============================
# LOAD MODEL KLASIFIKASI
# ===============================
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# ===============================
# LOAD HAND LANDMARKER (TASKS API)
# ===============================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)

# Label hasil klasifikasi
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D',4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Koneksi landmark manual (21 titik)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ===============================
# LOOPING REALTIME
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert ke format MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Deteksi tangan
    result = detector.detect(mp_image)

    if result.hand_landmarks:

        data_aux = []
        x_ = []
        y_ = []

        # Ambil tangan pertama
        hand_landmarks = result.hand_landmarks[0]

        # Ambil koordinat untuk normalisasi
        for landmark in hand_landmarks:
            x_.append(landmark.x)
            y_.append(landmark.y)

        # ===============================
        # GAMBAR TITIK LANDMARK
        # ===============================
        for landmark in hand_landmarks:
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # ===============================
        # GAMBAR GARIS KONEKSI
        # ===============================
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            x_start = int(hand_landmarks[start_idx].x * w)
            y_start = int(hand_landmarks[start_idx].y * h)

            x_end = int(hand_landmarks[end_idx].x * w)
            y_end = int(hand_landmarks[end_idx].y * h)

            cv2.line(frame, (x_start, y_start),
                     (x_end, y_end),
                     (255, 0, 0), 2)

        # ===============================
        # NORMALISASI DATA UNTUK MODEL
        # ===============================
        for landmark in hand_landmarks:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        # ===============================
        # PREDIKSI
        # ===============================
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Tampilkan teks hasil
        cv2.putText(frame,
                    predicted_character,
                    (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3)

    # Tampilkan frame
    cv2.imshow("Hand Detection", frame)

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# RELEASE
# ===============================
cap.release()
cv2.destroyAllWindows()
