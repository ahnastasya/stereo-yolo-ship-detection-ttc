import cv2
import os
import time

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)

#  FIX RESOLUSI BIAR GA LAG
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Kamera gagal")
    exit()

# ======================
# PATH
# ======================
base_path = "D:/Ahnastasya/KALIBRASI"
left_path = os.path.join(base_path, "left")
right_path = os.path.join(base_path, "right")

os.makedirs(left_path, exist_ok=True)
os.makedirs(right_path, exist_ok=True)

# ======================
# CHESSBOARD
# ======================
chessboard_size = (9, 6)

# ======================
# CONTROL
# ======================
count = 0
last_capture_time = 0
delay = 2  # lebih stabil (biar ga blur)

print("Auto capture aktif... tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    mid = w // 2

    # ======================
    # SPLIT (TANPA RESIZE)
    # ======================
    rawL = frame[:, :mid]
    rawR = frame[:, mid:]

    # ======================
    # GRAYSCALE
    # ======================
    grayL = cv2.cvtColor(rawL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rawR, cv2.COLOR_BGR2GRAY)

    # ======================
    # DETEKSI (LEBIH KUAT UNTUK FISHEYE)
    # ======================
    retL, cornersL = cv2.findChessboardCornersSB(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCornersSB(grayR, chessboard_size, None)

    # ======================
    # DISPLAY
    # ======================
    displayL = rawL.copy()
    displayR = rawR.copy()

    if retL:
        cv2.drawChessboardCorners(displayL, chessboard_size, cornersL, retL)

    if retR:
        cv2.drawChessboardCorners(displayR, chessboard_size, cornersR, retR)

    cv2.imshow("LEFT", displayL)
    cv2.imshow("RIGHT", displayR)

    # ======================
    # AUTO CAPTURE
    # ======================
    current_time = time.time()

    if retL and retR:
        if current_time - last_capture_time > delay:

            cv2.imwrite(os.path.join(left_path, f"img_{count}.jpg"), rawL)
            cv2.imwrite(os.path.join(right_path, f"img_{count}.jpg"), rawR)

            print(f"Captured {count}")
            count += 1
            last_capture_time = current_time

    # ======================
    # EXIT
    # ======================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()