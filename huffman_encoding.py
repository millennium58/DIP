import heapq
import numpy as np
import cv2


class HuffmanNode:
    def __init__(self, pixel, freq):
        self.pixel = pixel
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def calculate_frequencies(image):
    if image is None:
        raise ValueError("Error: Image not loaded. Please check the file path.")

    pixel_freq = {}
    for row in image:
        for pixel in row:
            if pixel in pixel_freq:
                pixel_freq[pixel] += 1
            else:
                pixel_freq[pixel] = 1
    return pixel_freq


def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(pixel, freq) for pixel, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def build_huffman_codes(node, code, mapping):
    if node is not None:
        if node.pixel is not None:
            mapping[node.pixel] = code
        build_huffman_codes(node.left, code + "0", mapping)
        build_huffman_codes(node.right, code + "1", mapping)


def compress_image(image, huffman_mapping):
    compressed_data = ""
    for row in image:
        for pixel in row:
            compressed_data += huffman_mapping[pixel]
    return compressed_data


def write_compressed_file(filename, compressed_data, huffman_mapping):
    with open(filename, "wb") as f:
        for pixel, code in huffman_mapping.items():
            f.write(bytes([pixel]))
            code_length = len(code)
            f.write(bytes([code_length]))
            f.write(code.encode("utf-8"))

        while len(compressed_data) % 8 != 0:
            compressed_data += "0"
        byte_array = bytearray(int(compressed_data[i:i + 8], 2) for i in range(0, len(compressed_data), 8))
        f.write(bytes(byte_array))


if __name__ == "__main__":
    image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original Image", image)

    pixel_freq = calculate_frequencies(image)

    huffman_tree = build_huffman_tree(pixel_freq)

    # Huffman codes
    huffman_mapping = {}
    build_huffman_codes(huffman_tree, "", huffman_mapping)

    compressed_data = compress_image(image, huffman_mapping)

    original_size = image.size
    compressed_size = len(compressed_data)

    saved_bits = original_size * 8 - compressed_size

    total_bits = compressed_size
    Lavg = (original_size / total_bits) * 100

    print(f"Total bits: {total_bits}")
    print(f"Total saved bits: {saved_bits}")
    print(f"Compression Ratio: {Lavg}%")

    compressed_filename = "compressed_image.txt"
    with open(compressed_filename, "w") as f:
        f.write(compressed_data)

    print(f"Compressed data saved in {compressed_filename}.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()