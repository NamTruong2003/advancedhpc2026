
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import cuda
import numpy as np
import time



@cuda.jit
def gray_image(image_flat, gray_flat, pixel_count):

    index = cuda.grid(1)

    if index < pixel_count:

        r = image_flat[index, 0]
        g = image_flat[index, 1]
        b = image_flat[index, 2]

        gray_flat[index] = (r + g + b) / 3.0


img = mpimg.imread('Image.jpg')

print(f"Data type: {type(img)}")
print(f"Image dimensions (Height x Width x Channels): {img.shape}")
print(f"Pixel data type: {img.dtype}")

height, width, channels = img.shape
pixel_count = img.shape[0] * img.shape[1]

img_float = img.astype(np.float32)
img_flattened = img_float.reshape(-1, 3)

print(f"Flattened shape: {img_flattened.shape} (Total pixels: {pixel_count:,})")
print(f"First pixel example (RGB): {img_flattened[0]}")

gray_flattened_cpu = np.zeros(pixel_count, dtype=np.float32)

start_time = time.time()

d_img_flattened = cuda.to_device(img_flattened)
d_gray_flattened = cuda.device_array(pixel_count, dtype=np.float32)

threads_per_block = 256
blocks_per_grid = (pixel_count + (threads_per_block - 1)) 

gray_image[blocks_per_grid, threads_per_block](
    d_img_flattened,
    d_gray_flattened,
    pixel_count
)

cuda.synchronize()

gray_flattened_cpu = d_gray_flattened.copy_to_host()

end_time = time.time()

print(f"Computation time: {(end_time - start_time) * 1000:.2f} ms")

gray_img = gray_flattened_cpu.reshape(height, width).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.title("Grayscale Image")
plt.imshow(gray_img, cmap='gray')
plt.axis('off')

plt.imsave('gray_image_gpu.png', gray_img, cmap='gray')

plt.show()

