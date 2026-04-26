import cv2
import numpy as np


IMAGE_PATH = "apple.jpg"
OUTPUT_CLUSTERED = "clustered_image.png"
OUTPUT_APPLES_MASK = "apples_mask.png"
OUTPUT_APPLES_SELECTED = "apples_selected.png"

K = 5


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

cv2.imwrite(OUTPUT_CLUSTERED, clustered_bgr)
cv2.imwrite(OUTPUT_APPLES_MASK, mask)
cv2.imwrite(OUTPUT_APPLES_SELECTED, apples_selected)

print("Clustering finished")
print("Saved files:")
print(OUTPUT_CLUSTERED)
print(OUTPUT_APPLES_MASK)
print(OUTPUT_APPLES_SELECTED)