import cv2
import numpy as np
import time
from ultralytics import YOLO

# ======================
# LOAD MODEL
# ======================
model = YOLO("C:/Users/LEGION/runs/detect/ship_yolov12m28/weights/best.pt")

# ======================
# LOAD CALIBRATION
# ======================
data = np.load("D:/Ahnastasya/KALIBRASI/stereo_params.npz")

K1 = data["camL"]
D1 = data["distL"]
K2 = data["camR"]
D2 = data["distR"]
R = data["R"]
T = data["T"]

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera failed")
    exit()

# ======================
# SGBM (TUNED)
# ======================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=150,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

rectify_done = False
SCALE_CORRECTION = 0.58

# ======================
# HISTORY
# ======================
depth_history = {}
time_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_full, w_full, _ = frame.shape
    mid = w_full // 2

    frameL = frame[:, :mid]
    frameR = frame[:, mid:]

    # ======================
    # RECTIFY
    # ======================
    if not rectify_done:
        h, w = frameL.shape[:2]

        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2, (w, h), R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0
        )

        mapLx, mapLy = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, P1, (w, h), cv2.CV_32FC1
        )
        mapRx, mapRy = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R2, P2, (w, h), cv2.CV_32FC1
        )

        rectify_done = True

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    # ======================
    # PREPROCESS
    # ======================
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    grayL = cv2.GaussianBlur(grayL, (5,5), 0)
    grayR = cv2.GaussianBlur(grayR, (5,5), 0)

    # ======================
    # DISPARITY
    # ======================
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    disp_vis = np.nan_to_num(disp)
    disp_vis = (disp_vis - disp_vis.min()) / (disp_vis.max() - disp_vis.min() + 1e-6)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    disp_vis = cv2.medianBlur(disp_vis, 5)
    disp_vis = cv2.bilateralFilter(disp_vis, 9, 75, 75)

    cv2.imshow("Disparity", disp_vis)

    # ======================
    # DEPTH
    # ======================
    points_3D = cv2.reprojectImageTo3D(disp, Q)

    # ======================
    # YOLO TRACK
    # ======================
    results = model.track(rectL, conf=0.3, persist=True, verbose=False)

    output = rectL.copy()

    for r in results:

        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, track_id, cls_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls_id = int(cls_id)

            class_name = model.names.get(cls_id, "obj")

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cy >= points_3D.shape[0] or cx >= points_3D.shape[1]:
                continue

            region = points_3D[max(0,cy-15):cy+15, max(0,cx-15):cx+15, 2]
            valid = region[np.isfinite(region)]

            if len(valid) < 10:
                continue

            Z = np.median(valid) * SCALE_CORRECTION

            if Z < 0.3 or Z > 5:
                continue

            # ======================
            # SMOOTH DEPTH
            # ======================
            if track_id not in depth_history:
                depth_history[track_id] = Z
            else:
                Z = 0.7 * depth_history[track_id] + 0.3 * Z
                depth_history[track_id] = Z

            # ======================
            # TIME TO COLLISION
            # ======================
            current_time = time.time()

            if track_id not in time_history:
                time_history[track_id] = (Z, current_time)
                velocity = 0
                ttc = -1
            else:
                prev_Z, prev_t = time_history[track_id]
                dt = current_time - prev_t

                if dt > 0:
                    velocity = (prev_Z - Z) / dt
                else:
                    velocity = 0

                time_history[track_id] = (Z, current_time)

                if velocity > 0.01:
                    ttc = Z / velocity
                else:
                    ttc = -1

            # ======================
            # COLOR INDICATOR
            # ======================
            if ttc > 0 and ttc < 3:
                color = (0, 0, 255)  # merah
                status = "DANGER"
            elif ttc >= 3 and ttc < 7:
                color = (0, 255, 255)  # kuning
                status = "WARNING"
            elif ttc >= 7:
                color = (0, 255, 0)  # hijau
                status = "SAFE"
            else:
                color = (255, 255, 255)
                status = ""

            # ======================
            # LABEL
            # ======================
            if ttc > 0:
                label = f"{class_name} {Z:.2f}m | {ttc:.1f}s"
            else:
                label = f"{class_name} {Z:.2f}m"

            # ======================
            # DRAW
            # ======================
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            cv2.putText(output, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.putText(output, f"ID:{track_id}", (x2 - 50, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if status != "":
                cv2.putText(output, status, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLO + Depth + TTC Warning", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()