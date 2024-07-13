import numpy as np
import cv2
import matplotlib.pyplot as plt

# Reading input image
input_image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)

# Saving the size of the input_image in pixels
# M: number of rows (height of the image)
# N: number of columns (width of the image)
M, N = input_image.shape

# Getting Fourier Transform of the input_image
FT_img = np.fft.fft2(input_image)

# Assign Cut-off Frequency
D0 = 10  # You can change this value accordingly

# Designing filter
u = np.arange(M)
idx = u > M // 2
u[idx] -= M
v = np.arange(N)
idy = v > N // 2
v[idy] -= N

# Creating a 2D grid of coordinates
V, U = np.meshgrid(v, u)

# Calculating Euclidean Distance
D = np.sqrt(U**2 + V**2)

# Comparing with the cut-off frequency and determining the filtering mask
H = 1.0 * (D > D0)

# Convolution between the Fourier Transformed image and the mask
G = H * FT_img

# Getting the resultant image by Inverse Fourier Transform
output_image = np.real(np.fft.ifft2(G))

# Displaying Input Image and Output Image
plt.subplot(2, 1, 1), plt.imshow(input_image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 1, 2), plt.imshow(output_image, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()
