import cv2
import numpy as np

# Load markers
marker0 = cv2.imread("marker_0.png")
marker1 = cv2.imread("marker_1.png")
marker2 = cv2.imread("marker_2.png")

markers = [marker0, marker1, marker2]

# Define A4 size at 300 DPI
a4_width, a4_height = 2480, 3508

# Create a white A4 canvas (3 channels)
a4_image = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255

# Define marker size on page (approx 1/3 width of A4 width)
marker_size = int(a4_width / 3)  # 826 pixels approx

# Positions for 2x2 square (top-left, top-right, bottom-left, bottom-right)
positions = [
    (50, 50),  # top-left
    (50, 1604),  # top-right
    (1604, 50),  # bottom-left
    (1604, 1604)  # bottom-right
]

# Place markers
for i, marker in enumerate(markers):
    # Resize marker
    resized_marker = cv2.resize(marker, (marker_size, marker_size), interpolation=cv2.INTER_NEAREST)
    y, x = positions[i]
    a4_image[y:y+marker_size, x:x+marker_size] = resized_marker

# Save final A4 image
cv2.imwrite("a4_markers_square.png", a4_image)