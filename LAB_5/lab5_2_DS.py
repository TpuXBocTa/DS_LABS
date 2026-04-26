import cv2
import numpy as np


IMAGE_PATH = "apple.jpg"

OUTPUT_MASK = "apples_mask_counting.png"
OUTPUT_RESULT = "apples_counted_result.png"

MIN_OBJECT_AREA = 40
MAX_OBJECT_AREA = 2500

MIN_CIRCULARITY = 0.25


image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image was not found")
    exit()

image = cv2.resize(image, (600, 600))

blurred = cv2.GaussianBlur(image, (5, 5), 0)

hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

lower_red_1 = np.array([0, 70, 60])
upper_red_1 = np.array([25, 255, 255])

lower_red_2 = np.array([165, 70, 60])
upper_red_2 = np.array([179, 255, 255])

mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

mask = cv2.bitwise_or(mask_1, mask_2)

kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

apple_count = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if area < MIN_OBJECT_AREA or area > MAX_OBJECT_AREA:
        continue

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if circularity < MIN_CIRCULARITY:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h

    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        continue

    apple_count += 1

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        str(apple_count),
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 0),
        1
    )

cv2.imwrite(OUTPUT_MASK, mask)
cv2.imwrite(OUTPUT_RESULT, result)

print("APPLE COUNTING RESULT")
print("=" * 30)
print(f"Detected apples: {apple_count}")
print(f"Saved mask: {OUTPUT_MASK}")
print(f"Saved result image: {OUTPUT_RESULT}")