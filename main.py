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

    for contour in contours: #find contours
        area = cv2.contourArea(contour)

        if area > 100:
            large_contours.append(contour)

    all_points = np.vstack(large_contours) #combine contours into one array

    rect = cv2.minAreaRect(all_points) #find rotated bounding box - finds smallest possible bounding box

    (center_x, center_y), (width, height), angle = rect #find angle of bounding box

    if width < height: #normalize angle
        angle = angle + 90

    print("fixed angle:", angle)

    height_img, width_img = img.shape[:2]

    center = (width_img // 2, height_img // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])

    new_width = int((height_img * sin) + (width_img * cos))
    new_height = int((height_img * cos) + (width_img * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated = cv2.warpAffine(
        img,
        rotation_matrix,
        (new_width, new_height)
    )

    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()