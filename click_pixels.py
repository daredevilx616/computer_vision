# click_pixels.py
import cv2

# Global list to store clicked points
clicked_points = []

def get_pixel_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")
        clicked_points.append((x, y))

# Load the image
image_path = "your_image.jpg"  # Replace with your actual image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Create a named window with the exact image size
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", image.shape[1], image.shape[0])

# Set mouse callback
cv2.setMouseCallback("Image", get_pixel_coordinates)

# Show image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"All clicked points: {clicked_points}")
