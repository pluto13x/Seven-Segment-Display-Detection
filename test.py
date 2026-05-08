import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("public dataset\images/13.png")

# Split BGR channels
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

# Detect glowing red pixels
red_mask = (r > 200) & (g < 80) & (b < 80)

# Detect glowing green pixels
green_mask = (g > 200) & (r < 80) & (b < 80)

# Detect glowing blue pixels
blue_mask = (b > 200) & (r < 80) & (g < 80)

# Combine all masks
led_mask = red_mask | green_mask | blue_mask

# Convert boolean masks to uint8 images
red_img = red_mask.astype(np.uint8) * 255
green_img = green_mask.astype(np.uint8) * 255
blue_img = blue_mask.astype(np.uint8) * 255
led_img = led_mask.astype(np.uint8) * 255

# Show results
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(red_img, cmap="gray")
plt.title("Red Pixels")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(green_img, cmap="gray")
plt.title("Green Pixels")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(blue_img, cmap="gray")
plt.title("Blue Pixels")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(led_img, cmap="gray")
plt.title("Combined LED Mask")
plt.axis("off")

plt.tight_layout()
plt.show()