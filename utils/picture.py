import cv2
import csv
import matplotlib.pyplot as plt

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked coordinates: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', img)

try:
    img = cv2.imread('../image/image.png')
    if img is None:
        raise FileNotFoundError("Image not found or could not be loaded. Please check the file path.")
except FileNotFoundError as e:
    print(e)
    exit()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

cv2.imshow('Image', img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\nAll clicked coordinate points:", points)

with open('clicked_points.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['name', 'x', 'y'])
    for idx, (x, y) in enumerate(points, start=1):
        writer.writerow([idx, x, y])

print("Coordinate points have been saved to clicked_points.csv")
