import os
import cv2
import pickle
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "DATA")
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

data = []
labels = []

print("Memulai ekstraksi data...")

with HandLandmarker.create_from_options(options) as landmarker:

    for label in os.listdir(DATA_DIR):

        folder_path = os.path.join(DATA_DIR, label)


        if not os.path.isdir(folder_path):
            continue

        print("Memproses folder:", label)

        for img_name in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)


            if img is None:
                print("Gambar gagal dibaca:", img_path)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img_rgb
            )

            detection_result = landmarker.detect(mp_image)

            if detection_result.hand_landmarks:

                for hand_landmarks in detection_result.hand_landmarks:

                    x_ = []
                    y_ = []
                    data_aux = []

                    for landmark in hand_landmarks:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    for landmark in hand_landmarks:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                    data.append(data_aux)
                    labels.append(label)

print("Jumlah data berhasil diekstraksi:", len(data))


with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Selesai! File data.pickle berhasil dibuat.")
