import numpy as np
import cv2

# Reading input image
input_image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)



M, N = input_image.shape

# Getting Fourier Transform of the input_image
FT_img = np.fft.fft2(input_image)

# Assign the order value
n = 2  # You can change this value accordingly

D0 = 30
 # You can change this value accordingly

# Designing filter
u = np.arange(M)
v = np.arange(N)
idx = np.where(u > M / 2)
u[idx] -= M
idy = np.where(v > N / 2)
v[idy] -= N

# Creating a meshgrid
V, U = np.meshgrid(v, u)

# Calculating Euclidean Distance
D = np.sqrt(U**2 + V**2)

# Determining the filtering mask
H = 1 / (1 + (D / D0)**(2 * n))

# Convolution between the Fourier Transformed image and the mask
G = H * FT_img

# Getting the resultant image by Inverse Fourier Transform
output_image = np.real(np.fft.ifft2(G))

# Displaying Input Image and Output Image
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', np.uint8(output_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
