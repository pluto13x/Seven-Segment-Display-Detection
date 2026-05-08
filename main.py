import os
import cv2
import matplotlib.pyplot as plt

folder = "public dataset/images"
images = []

for filename in os.listdir(folder): #load images
    path = os.path.join(folder, filename)
    img = cv2.imread(path)
    
    # Split BGR channels
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # Detect glowing pixels
    red_mask = (r > 200) & (g < 80) & (b < 80)
    green_mask = (g > 200) & (r < 80) & (b < 80)
    blue_mask = (b > 200) & (r < 80) & (g < 80)

    # Combine all masks
    led_mask = red_mask | green_mask | blue_mask
    images.append(led_mask.astype('uint8') * 255)
    plt.figure(figsize=(6, 6))
    plt.imshow(led_mask.astype('uint8') * 255, cmap="gray")
    plt.title(filename)
    plt.axis("off")
    plt.show()
