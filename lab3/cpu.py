import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import cuda
import numpy as np
import time

img = mpimg.imread('Image.jpg') 

print(f"Data type: {type(img)}")
print(f"Image dimensions (Height x Width x Channels): {img.shape}")
print(f"Pixel data type: {img.dtype}")

height, width, channels = img.shape
pixel_count = img.shape[0] * img.shape[1]
img_flattened = img.reshape(-1, 3)

print(f"Flattened shape: {img_flattened.shape} (Total pixels: {pixel_count:,})")
print(f"First pixel example (RGB): {img_flattened[0]}")

gray_flattened = np.zeros(pixel_count,dtype=np.float32)

start_time  = time.time()

for index in range(pixel_count):
    r = img_flattened[index,0] / 3.0
    g = img_flattened[index,1] / 3.0
    b = img_flattened[index,2] / 3.0
    gray_flattened[index] = r + g + b
end_time = time.time()

print(f"time to compute { (end_time -start_time)*1000:.2f}")

gray_img = gray_flattened.reshape(height, width).astype(np.uint8)
plt.figure(figsize=(6, 6))
plt.title("Gray image ")
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.imsave('gray_image_cpu.png', gray_img, cmap='gray')
plt.show()