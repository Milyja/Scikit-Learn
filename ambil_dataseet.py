import cv2
import os

# Label
label = input("Masukkan label (0 = Tutup, 1 = Buka): ")

folder_path = f"DATA/{label}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

cap = cv2.VideoCapture(0)

counter = 0

print("Tekan 's' untuk simpan gambar")
print("Tekan 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Ambil Dataset", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        file_path = os.path.join(folder_path, f"{counter}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Gambar disimpan: {file_path}")
        counter += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
