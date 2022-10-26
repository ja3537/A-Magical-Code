import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq
    
    def __eq__(self, other):
        return self.freq == other.freq

def make_encoding(frequencies):
    """
    frequencies is a dictionary mapping from a character in the English alphabet to its frequency.
    Returns a dictionary mapping from a character to its Huffman encoding.
    """
    huffman_encoding = {}
    heap = []

    for char, freq in frequencies.items():
        heapq.heappush(heap, Node(char, freq))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        node = Node(None, left.freq + right.freq)
        node.left = left
        node.right = right
        heapq.heappush(heap, node)

    root = heapq.heappop(heap)

    def traverse(node, encoding):
        if node.char:
            huffman_encoding[node.char] = encoding
        else:
            traverse(node.left, encoding + '0')
            traverse(node.right, encoding + '1')

    traverse(root, '')

    return huffman_encoding

def encode_message(message, encoding):
    """
    Encodes a message using the given encoding.
    """
    processed_message = message.replace(' ', '')

    encoded_message = []
    for char in processed_message:
        encoded_message.append(encoding[char])
    return ''.join(encoded_message)

def decode_message(encoded_message, encoding):
    """
    Decodes a message using the given encoding.
    """
    decoded_message = ''
    while encoded_message:
        for char, code in encoding.items():
            if encoded_message.startswith(code):
                decoded_message += char
                encoded_message = encoded_message[len(code):]
    return decoded_message

class Agent:
    def __init__(self):
        pass

    def encode(self, message):
        return list(range(52))

    def decode(self, deck):
        return "NULL"
    