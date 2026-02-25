import os
import cv2
import time

DATA_DIR = r'C:\Users\user\Documents\Datasheet tubuh\Praktikum\DATA'
NUMBER_OF_CLASSES = 2
DATASET_SIZE = 100
DELAY_START = 10


os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for j in range(NUMBER_OF_CLASSES):

    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'\nMengumpulkan data untuk kelas {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f'Class {j} | Press Q to start',
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow('Dataset Capture', frame)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break


    for i in range(DELAY_START, 0, -1):
        ret, frame = cap.read()

        cv2.putText(
            frame,
            f'Starting in {i}...',
            (120, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4
        )

        cv2.imshow('Dataset Capture', frame)
        cv2.waitKey(1000)


    counter = len([
        f for f in os.listdir(class_dir)
        if f.endswith('.jpg')
    ])

    print(f'Mulai dari gambar ke-{counter}')

    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f'Class {j} | Image {counter}/{DATASET_SIZE}',
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow('Dataset Capture', frame)
        cv2.waitKey(10)

        cv2.imwrite(
            os.path.join(class_dir, f'{counter}.jpg'),
            frame
        )

        counter += 5

    print(f'Kelas {j} selesai.')


cap.release()
cv2.destroyAllWindows()

print('\nPengambilan dataset selesai.')