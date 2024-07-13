import cv2
import numpy as np
import heapq


class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob  # Store the probability as a tuple (frequency, symbol)
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(frequency):
    heap = [Node((freq, symbol), symbol) for symbol, freq in enumerate(frequency) if freq > 0]

    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged_prob = (left.prob[0] + right.prob[0], left.prob[1] + right.prob[1])
        new_node = Node(merged_prob, left.symbol + right.symbol, left, right)
        heapq.heappush(heap, new_node)

    return heap[0]


def generate_huffman_codes(node, current_code="", huffman_codes=None, code_lengths=None):
    if huffman_codes is None:
        huffman_codes = {}

    if code_lengths is None:
        code_lengths = {}

    if node is None:
        return huffman_codes, code_lengths

    if node.left is None and node.right is None:
        huffman_codes[node.prob[1]] = current_code
        code_lengths[node.prob[1]] = len(current_code)
        return huffman_codes, code_lengths  # Return early for leaf nodes.

    huffman_codes, code_lengths = generate_huffman_codes(node.left, current_code + "0", huffman_codes, code_lengths)
    huffman_codes, code_lengths = generate_huffman_codes(node.right, current_code + "1", huffman_codes, code_lengths)

    return huffman_codes, code_lengths


def cal_frequency(input_image):
    return np.bincount(input_image.flatten(), minlength=256)  # Assuming 8-bit grayscale.


def encode_image(input_image, huffman_codes, code_lengths):
    height, width = input_image.shape
    encoded_data = ''
    for i in range(height):
        for j in range(width):
            pixel_value = input_image[i, j]
            encoded_data += huffman_codes[pixel_value][:code_lengths[pixel_value]]
    return encoded_data


def huffman_encoding(input_image):
    frequency = cal_frequency(input_image)
    huffman_tree = build_huffman_tree(frequency)
    huffman_codes, code_lengths = generate_huffman_codes(huffman_tree)

    encoded_data = encode_image(input_image, huffman_codes, code_lengths)

    return encoded_data, huffman_tree


def decode_data(encoded_data, huffman_tree):
    decoded_image = []
    current_node = huffman_tree

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.left is None and current_node.right is None:
            decoded_image.append(current_node.prob[1])
            current_node = huffman_tree

    return np.array(decoded_image).reshape(input_image.shape)


if __name__ == '__main__':
    input_image = cv2.imread('/home/phoenix/Documents/wpl/hudai.jpg', cv2.IMREAD_GRAYSCALE)
    encoded_data, huffman_tree = huffman_encoding(input_image)

    print("Huffman Codes:")
    huffman_codes, code_lengths = generate_huffman_codes(huffman_tree)
    for pixel, code in sorted(huffman_codes.items(), key=lambda x: code_lengths[x[0]]):
        print(f"Pixel: {pixel}, Code: {code}")

    original_image_size = input_image.shape[0] * input_image.shape[1] * 8  # Assuming 8 bits per pixel for grayscale.
    encoded_data_size = len(encoded_data)
    compression_ratio = original_image_size / encoded_data_size

    print(f"Original Image Size: {original_image_size} bits")
    print(f"Encoded Data Size: {encoded_data_size} bits")
    print(f"Compression Ratio: {compression_ratio:.2f}")

    decoded_image = decode_data(encoded_data, huffman_tree)
    cv2.imwrite('/home/phoenix/Documents/wpl/decoded_hudai.jpg', decoded_image)
