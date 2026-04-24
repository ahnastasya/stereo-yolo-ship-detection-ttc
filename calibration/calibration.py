import cv2
import numpy as np
import os

# ======================
# PATH
# ======================
left_path = r"D:\Ahnastasya\KALIBRASI\left"
right_path = r"D:\Ahnastasya\KALIBRASI\right"

images_left = sorted([f for f in os.listdir(left_path) if f.endswith(".jpg")])
images_right = sorted([f for f in os.listdir(right_path) if f.endswith(".jpg")])

print(f"Total Left: {len(images_left)}")
print(f"Total Right: {len(images_right)}")

# ======================
# CHESSBOARD
# ======================
chessboard_size = (9, 6)
square_size = 0.02  # meter (sesuaikan)

# ======================
# OBJECT POINT
# ======================
objp = np.zeros((1, chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

obj_points = []
img_points_L = []
img_points_R = []

image_size = None
valid = 0

# ======================
# LOOP
# ======================
for i in range(min(len(images_left), len(images_right))):

    imgL = cv2.imread(os.path.join(left_path, images_left[i]))
    imgR = cv2.imread(os.path.join(right_path, images_right[i]))

    if imgL is None or imgR is None:
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = grayL.shape[::-1]

    # DETEKSI KUAT
    retL, cornersL = cv2.findChessboardCornersSB(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCornersSB(grayR, chessboard_size, None)

    print(f"{images_left[i]} -> L:{retL}, R:{retR}")

    if retL and retR:

        # FILTER DATA JELEK
        if cornersL.shape != cornersR.shape:
            continue

        if cornersL.shape[0] < 40:
            continue

        # FORMAT FISHEYE WAJIB
        obj_points.append(objp.astype(np.float64))
        img_points_L.append(cornersL.reshape(1, -1, 2).astype(np.float64))
        img_points_R.append(cornersR.reshape(1, -1, 2).astype(np.float64))

        valid += 1

print(f"\n Valid pairs (filtered): {valid}")

if valid < 10:
    print(" Data kurang / terlalu banyak gambar jelek")
    exit()

# ======================
# INIT MATRIX
# ======================
K_L = np.zeros((3,3))
D_L = np.zeros((4,1))

K_R = np.zeros((3,3))
D_R = np.zeros((4,1))

# ======================
# SINGLE CALIBRATION
# ======================
retL, camL, distL, _, _ = cv2.fisheye.calibrate(
    obj_points, img_points_L, image_size,
    K_L, D_L, None, None,
    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
)

retR, camR, distR, _, _ = cv2.fisheye.calibrate(
    obj_points, img_points_R, image_size,
    K_R, D_R, None, None,
    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
)

print(" Fisheye calibration OK")

# ======================
# STEREO
# ======================
print(" Data siap stereo:", len(obj_points))

criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)

try:
    # FIX DI SINI (9 OUTPUT)
    ret, camL, distL, camR, distR, R, T, _, _ = cv2.fisheye.stereoCalibrate(
        obj_points,
        img_points_L,
        img_points_R,
        camL,
        distL,
        camR,
        distR,
        image_size,
        criteria=criteria,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC
    )

    print(" Stereo fisheye DONE")

except cv2.error as e:
    print(" Stereo gagal:", e)
    print(" Solusi:")
    print(" - Hapus 5-10 gambar paling jelek")
    print(" - Hindari sudut ekstrim banget")
    exit()

# ======================
# SAVE
# ======================
np.savez("stereo_params.npz",
         camL=camL, distL=distL,
         camR=camR, distR=distR,
         R=R, T=T)

baseline = np.linalg.norm(T)
print(f" Baseline: {baseline:.4f} meter")

print("\n SELESAI! File tersimpan: stereo_params.npz")