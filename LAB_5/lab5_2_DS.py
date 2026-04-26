import cv2
import numpy as np


IMAGE_PATH = "apple.jpg"

OUTPUT_CLUSTERED = "clustered_image.png"
OUTPUT_APPLES_MASK = "apples_mask.png"
OUTPUT_APPLES_SELECTED = "apples_selected.png"
OUTPUT_RESULT = "apples_counted_result.png"

K = 5

MIN_OBJECT_AREA = 80
MAX_OBJECT_AREA = 8000
MIN_CIRCULARITY = 0.20

SPLIT_CLUSTER_AREA = 500
MIN_APPLE_RADIUS = 5
PEAK_THRESHOLD = 0.45
MIN_FILL_RATIO = 0.20


image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image was not found")
    exit()

image = cv2.resize(image, (600, 600))
blurred = cv2.GaussianBlur(image, (5, 5), 0)

lab_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

pixels = lab_image.reshape((-1, 3))
pixels = np.float32(pixels)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
labels = labels.flatten()

clustered_pixels = centers[labels]
clustered_lab = clustered_pixels.reshape(lab_image.shape)
clustered_bgr = cv2.cvtColor(clustered_lab, cv2.COLOR_LAB2BGR)

centers_bgr = cv2.cvtColor(
    centers.reshape(1, K, 3),
    cv2.COLOR_LAB2BGR
).reshape(K, 3)

centers_hsv = cv2.cvtColor(
    centers_bgr.reshape(1, K, 3),
    cv2.COLOR_BGR2HSV
).reshape(K, 3)

apple_clusters = []

for i in range(K):
    h, s, v = centers_hsv[i]
    b, g, r = centers_bgr[i]

    if ((h <= 25 or h >= 165) and s > 60 and v > 80 and r > g and r > b):
        apple_clusters.append(i)

print("Apple color clusters:", apple_clusters)

mask = np.zeros(labels.shape, dtype=np.uint8)

for cluster_index in apple_clusters:
    mask[labels == cluster_index] = 255

mask = mask.reshape(image.shape[:2])

kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

apples_selected = cv2.bitwise_and(image, image, mask=mask)


def split_cluster_into_apples(cluster_mask, offset_x, offset_y):
    apples = []

    distance = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, 5)
    max_distance = distance.max()

    if max_distance <= 0:
        return apples

    peaks = np.zeros(distance.shape, dtype=np.uint8)
    peaks[distance >= PEAK_THRESHOLD * max_distance] = 255

    small_kernel = np.ones((3, 3), np.uint8)
    peaks = cv2.erode(peaks, small_kernel, iterations=1)

    num_labels, components = cv2.connectedComponents(peaks)

    for component_index in range(1, num_labels):
        ys, xs = np.where(components == component_index)

        if len(xs) == 0:
            continue

        component_distances = distance[ys, xs]
        best_index = np.argmax(component_distances)

        center_y = ys[best_index]
        center_x = xs[best_index]
        radius = int(distance[center_y, center_x])

        if radius < MIN_APPLE_RADIUS:
            continue

        circle_mask = np.zeros(cluster_mask.shape, dtype=np.uint8)

        cv2.circle(
            circle_mask,
            (center_x, center_y),
            radius,
            255,
            -1
        )

        overlap = cv2.bitwise_and(cluster_mask, circle_mask)
        filled_pixels = cv2.countNonZero(overlap)

        circle_area = np.pi * radius * radius
        fill_ratio = filled_pixels / circle_area

        if fill_ratio < MIN_FILL_RATIO:
            continue

        apples.append((offset_x + center_x, offset_y + center_y, radius))

    filtered_apples = []

    for apple in sorted(apples, key=lambda item: item[2], reverse=True):
        ax, ay, ar = apple
        duplicate = False

        for saved in filtered_apples:
            sx, sy, sr = saved
            center_distance = np.sqrt((ax - sx) ** 2 + (ay - sy) ** 2)

            if center_distance < 0.7 * max(ar, sr):
                duplicate = True
                break

        if not duplicate:
            filtered_apples.append(apple)

    return filtered_apples


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

    x, y, w, h = cv2.boundingRect(contour)

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    aspect_ratio = w / h

    local_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - [x, y]

    cv2.drawContours(local_mask, [shifted_contour], -1, 255, -1)

    found_apples = []

    if area >= SPLIT_CLUSTER_AREA:
        found_apples = split_cluster_into_apples(local_mask, x, y)

    if len(found_apples) >= 2:
        for cx, cy, radius in found_apples:
            apple_count += 1

            cv2.circle(result, (cx, cy), radius, (255, 0, 0), 2)
            cv2.putText(
                result,
                str(apple_count),
                (cx - radius, cy - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )
    else:
        if circularity < MIN_CIRCULARITY:
            continue

        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        apple_count += 1

        center_x = x + w // 2
        center_y = y + h // 2
        radius = int(np.sqrt(area / np.pi))

        cv2.circle(result, (center_x, center_y), radius, (255, 0, 0), 2)
        cv2.putText(
            result,
            str(apple_count),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1
        )

cv2.imwrite(OUTPUT_CLUSTERED, clustered_bgr)
cv2.imwrite(OUTPUT_APPLES_MASK, mask)
cv2.imwrite(OUTPUT_APPLES_SELECTED, apples_selected)
cv2.imwrite(OUTPUT_RESULT, result)

print("Clustering and counting finished")
print("Saved files:")
print(OUTPUT_CLUSTERED)
print(OUTPUT_APPLES_MASK)
print(OUTPUT_APPLES_SELECTED)
print(OUTPUT_RESULT)
print("Detected apples:", apple_count)