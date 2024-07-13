import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading input image
input_image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)

# Saving the size of the input_image in pixels
M, N = input_image.shape

# Getting Fourier Transform of the input_image
FT_img = np.fft.fft2(input_image)

# Assign the order value
n = 2  # You can change this value accordingly

# Assign Cut-off Frequency
D0 = 70  # You can change this value accordingly

# Designing filter
# Designing filter
u = np.arange(M)
v = np.arange(N)
idx = u > M / 2
idy = v > N / 2
u[idx] -= M
v[idy] -= N


# Creating a meshgrid
V, U = np.meshgrid(v, u)

# Calculating Euclidean Distance
D = np.sqrt(U**2 + V**2)

# Determining the filtering mask
H = 1 / (1 + (D0 / D)**(2 * n))

# Convolution between the Fourier Transformed image and the mask
G = H * FT_img

# Getting the resultant image by Inverse Fourier Transform
output_image = np.abs(np.fft.ifft2(G))

# Displaying Input Image and Output Image
plt.subplot(2, 1, 1), plt.imshow(input_image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 1, 2), plt.imshow(output_image, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()
