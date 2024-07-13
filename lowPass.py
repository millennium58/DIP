import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading input image
input_image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)

# Saving the size of the input_image in pixels
M, N = input_image.shape

# Assign Cut-off Frequency
D0 = 30  # You can change this value accordingly

# Designing filter
u = np.arange(M)
idx = np.where(u > M // 2)
u[idx] = u[idx] - M
v = np.arange(N)
idy = np.where(v > N // 2)
v[idy] = v[idy] - N

# Creating a meshgrid
V, U = np.meshgrid(v, u)

# Calculating Euclidean Distance
D = np.sqrt(U**2 + V**2)

# Comparing with the cut-off frequency and determining the filtering mask
H = np.double(D <= D0)

# Perform manual Fourier Transform
'''FT_img = np.zeros_like(input_image, dtype=complex)
for x in range(M):
    for y in range(N):
        sum_val = 0
        for u in range(M):
            for v in range(N):
                sum_val += input_image[u, v] * np.exp(-1j * 2 * np.pi * (x * u / M + y * v / N))
        FT_img[x, y] = sum_val'''

FT_img = np.fft.fft2(input_image)

# Perform filtering in the frequency domain
G = H * FT_img

# Perform manual Inverse Fourier Transform
'''
output_image = np.zeros_like(input_image, dtype=float)
for x in range(M):
    for y in range(N):
        sum_val = 0
        for u in range(M):
            for v in range(N):
                sum_val += G[u, v] * np.exp(1j * 2 * np.pi * (x * u / M + y * v / N))
        output_image[x, y] = np.real(sum_val)
'''
output_image = np.real(np.fft.ifft2(G))
# Displaying Input Image and Output Image using OpenCV
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', np.uint8(output_image))
cv2.imwrite('C:\\Users\\USER\\Pictures\\save.png', np.uint8(output_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
