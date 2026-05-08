import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

folder = "public dataset/images"
images = []

for filename in os.listdir(folder): #load images
    #create mask
    path = os.path.join(folder, filename)
    print(path)
    img = cv2.imread(path)
    
    # Split BGR channels
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # Detect glowing pixels
    red_mask = (r > 160) & (g < 80) & (b < 80)
    green_mask = (g > 80) & (r < 80) & (b < 80)
    blue_mask = (b > 180) & (r < 80) & (g < 80)

    # Combine all masks
    led_mask = red_mask | green_mask | blue_mask
    led_img = led_mask.astype('uint8') * 255

    contours, _ = cv2.findContours(
        led_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    large_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            large_contours.append(contour)

    output = cv2.cvtColor(led_img, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(
        output,
        large_contours,
        -1,
        (0, 255, 0),
        3
    )

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()