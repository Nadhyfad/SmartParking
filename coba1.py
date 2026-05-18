import cv2
from ultralytics import YOLO

# ── KONFIGURASI ──────────────────────────────────────────────
VIDEO_SOURCE = 0          # 0 = webcam, atau ganti path video/RTSP CCTV
TOTAL_CAPACITY = 100      # Sesuaikan kapasitas total slot parkir
CONFIDENCE = 0.4          # Threshold confidence deteksi
# ─────────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")  # Download otomatis saat pertama kali dijalankan

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE, classes=[3])  # class 3 = motorcycle (COCO)

    detections = results[0].boxes
    occupied = len(detections)
    empty = max(TOTAL_CAPACITY - occupied, 0)
    availability_pct = (empty / TOTAL_CAPACITY) * 100

    # Gambar bounding box tiap motor terdeteksi
    for box in detections.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Panel info di pojok kiri atas
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Kapasitas Total : {TOTAL_CAPACITY}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Slot Terisi     : {occupied}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
    cv2.putText(frame, f"Slot Kosong     : {empty}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 80), 2)
    cv2.putText(frame, f"Ketersediaan    : {availability_pct:.1f}%", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

    cv2.imshow("SmartParking AI - Motor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()