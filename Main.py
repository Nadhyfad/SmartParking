import cv2
import numpy as np

# ==========================
# BACA GAMBAR
# ==========================
image_path = "parkir.jpg"  # ganti dengan nama file gambarmu

img = cv2.imread(image_path)

if img is None:
    print("ERROR: Gambar tidak ditemukan!")
    exit()

# Resize agar konsisten
img = cv2.resize(img, (1280, 720))

# Salinan untuk hasil akhir
result = img.copy()

# ==========================
# PREPROCESSING
# ==========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kurangi noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Deteksi tepi
edges = cv2.Canny(
    blur,
    threshold1=50,
    threshold2=150
)

# ==========================
# DETEKSI GARIS
# ==========================
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=120,
    maxLineGap=20
)

# ==========================
# FILTER GARIS PARKIR
# ==========================
if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line[0]

        # Hitung sudut garis
        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        # Ambil garis vertikal & horizontal
        if (
            abs(angle) > 70 or
            abs(angle) < 20
        ):

            cv2.line(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

# ==========================
# TAMPILKAN HASIL
# ==========================
cv2.imshow("Original", img)
cv2.imshow("Edge Detection", edges)
cv2.imshow("Parking Line Detection", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
