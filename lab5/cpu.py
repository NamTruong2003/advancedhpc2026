
import matplotlib.image as mpimg
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
def flip_kernel(kernel):
    return kernel[::-1, ::-1]

def getNeighborhood(x,y,host,size):
    e = size // 2
    nb = [[0 for _ in range(size)] for _ in range(size)]
    rows,cols = host.shape
    for a in range(-e,e+1):
        for b in range(-e,e+1):
            if(x + a) < 0 or (y + b)< 0 or (x + a) > rows -1 or (y + b) > cols - 1:
                continue
            nb[a + e][b+e] = host[x+a,y+b]
    return nb
def element_wise_multi(A,B):
    total = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            total += A[i][j] * B[i][j]
    return total
def convolution(kernel,image):
    kernel = flip_kernel(kernel)
    a = len(image)
    b = len(image[0])
    c = 3
    result = np.zeros((a, b, c), dtype=np.float32)
    for n in range(c):
        matrix2D = image[:,:,n]
        for w in range(a):
            for h in range(b):
            
                nb = getNeighborhood(w,h,matrix2D,7)
                result[w][h][n] = element_wise_multi(kernel,nb)
    return  np.clip(result, 0, 255).astype(np.uint8)
img = mpimg.imread('Image.jpg')
height, width, channels = img.shape
pixel_count = height * width

raw_kernel =  np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
],dtype=np.float32)
kernel= raw_kernel/ np.sum(raw_kernel)
blur_image = convolution(kernel=kernel,image=img)
plt.figure(figsize=(6, 6))
plt.title("Blur image ")
plt.imshow(blur_image)
plt.axis('off')
plt.imsave('blur_image_cpu.png', blur_image)
plt.show()