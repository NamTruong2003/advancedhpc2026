
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import cuda
import numpy as np
import time



@cuda.jit
def gray_image(image_flat, gray_flat, height,width):

    x,y = cuda.grid(2)

    if x < width and y < height:

        r = image_flat[y,x, 0]
        g = image_flat[y,x, 1]
        b = image_flat[y,x, 2]

        gray_flat[y,x] = (r + g + b) / 3
img = mpimg.imread('Image2.jpg')
print(f"Data type: {type(img)}")
print(f"Image dimensions (Height x Width x Channels): {img.shape}")
print(f"Pixel data type: {img.dtype}")
height, width, channels = img.shape
pixel_count = img.shape[0] * img.shape[1]
img_float = img.astype(np.float32)
gray_flattened_cpu = np.zeros((height,width), dtype=np.float32)
start_time = time.time()
d_img_flattened = cuda.to_device(img_float)
d_gray_flattened = cuda.device_array((height,width), dtype=np.float32)
block_size = (32,32)
grid_size = (width //block_size[0]+1,height // block_size[1] + 1)
gray_image[grid_size, block_size](
    d_img_flattened,
    d_gray_flattened,
    height,
    width
)
cuda.synchronize()
gray_flattened_cpu = d_gray_flattened.copy_to_host()
end_time = time.time()
print(f"Computation time: {(end_time - start_time) * 1000:.2f} ms")
plt.figure(figsize=(6, 6))
plt.title("Grayscale Image")
plt.imshow(gray_flattened_cpu, cmap='gray')
plt.axis('off')
plt.imsave('gray_image_gpu.png', gray_flattened_cpu, cmap='gray')


plt.show()

blocks = [(8,8),(16,8),(8,16),(16,16),(32,16),(16,32),(32,32)]
timeList = []
def grayTranfer(block,w,h):
    x = w // block[0] + 1
    y = h // block[1] + 1
    grid_size = (x,y)
    start = time.time()
    gray_image[grid_size,block](d_img_flattened,
        d_gray_flattened,
        height,
        width)
    cuda.synchronize()
    end = time.time()
    return end - start
def test(blocks):
    for block in blocks:
        total_time = 0 
        for i in range(1000):
            
            total_time += grayTranfer(block,width, height)
        timeList.append(f"{total_time:.2f}")
        
test(blocks)
print(timeList)