import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_low_pass_filter(shape, d0):
    """
    Create a Gaussian low-pass filter of order 2.
    
    Args:
    shape (tuple): Shape of the filter.
    d0 (float): Cutoff frequency.
    
    Returns:
    np.ndarray: The Gaussian low-pass filter.
    """
    rows, cols = shape
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), dtype=np.float32)
    
    for u in range(rows):
        for v in range(cols):
            du = u - center_x
            dv = v - center_y
            d = np.sqrt(du**2 + dv**2)
            filter[u, v] = np.exp(-(d**2) / (2 * (d0**2)))
    
    return filter

# Load the input image
image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)

# Define the cutoff frequency (D0 value)
d0 = 230

# Apply the Gaussian low-pass filter
filter_shape = image.shape
h = gaussian_low_pass_filter(filter_shape, d0)
filtered_image = np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(image)) * h)
filtered_image = np.abs(np.fft.ifft2(filtered_image))

# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title(f'Filtered Image (D0={d0})')
plt.axis('off')

plt.show()
