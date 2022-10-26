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


# Function to find the Checksum of Sent Message
def findChecksum(SentMessage, k):

    # Dividing sent message in packets of k bits.
    c1 = SentMessage[0:k]
    c2 = SentMessage[k:2*k]
    c3 = SentMessage[2*k:3*k]
    c4 = SentMessage[3*k:4*k]

    # Calculating the binary sum of packets
    Sum = bin(int(c1, 2)+int(c2, 2)+int(c3, 2)+int(c4, 2))[2:]

    # Adding the overflow bits
    if(len(Sum) > k):
        x = len(Sum)-k
        Sum = bin(int(Sum[0:x], 2)+int(Sum[x:], 2))[2:]
    if(len(Sum) < k):
        Sum = '0'*(k-len(Sum))+Sum

    # Calculating the complement of sum
    Checksum = ''
    for i in Sum:
        if(i == '1'):
            Checksum += '0'
        else:
            Checksum += '1'
    return Checksum

# Function to find the Complement of binary addition of
# k bit packets of the Received Message + Checksum
def checkReceiverChecksum(ReceivedMessage, k, Checksum):

    # Dividing sent message in packets of k bits.
    c1 = ReceivedMessage[0:k]
    c2 = ReceivedMessage[k:2*k]
    c3 = ReceivedMessage[2*k:3*k]
    c4 = ReceivedMessage[3*k:4*k]

    # Calculating the binary sum of packets + checksum
    ReceiverSum = bin(int(c1, 2)+int(c2, 2)+int(Checksum, 2) +
                    int(c3, 2)+int(c4, 2)+int(Checksum, 2))[2:]

    # Adding the overflow bits
    if(len(ReceiverSum) > k):
        x = len(ReceiverSum)-k
        ReceiverSum = bin(int(ReceiverSum[0:x], 2)+int(ReceiverSum[x:], 2))[2:]

    # Calculating the complement of sum
    ReceiverChecksum = ''
    for i in ReceiverSum:
        if(i == '1'):
            ReceiverChecksum += '0'
        else:
            ReceiverChecksum += '1'
    return ReceiverChecksum
    


# Driver Code
SentMessage = "10010101011000111001010011101100"
k = 8
#ReceivedMessage = "10000101011000111001010011101101"
ReceivedMessage = "10010101011000111001010011101100"
# Calling the findChecksum() function
Checksum = findChecksum(SentMessage, k)

# Calling the checkReceiverChecksum() function
ReceiverChecksum = checkReceiverChecksum(ReceivedMessage, k, Checksum)

# Printing Checksum
print("SENDER SIDE CHECKSUM: ", Checksum)
print("RECEIVER SIDE CHECKSUM: ", ReceiverChecksum)
finalsum=bin(int(Checksum,2)+int(ReceiverChecksum,2))[2:]

# Finding the sum of checksum and received checksum
finalcomp=''
for i in finalsum:
    if(i == '1'):
        finalcomp += '0'
    else:
        finalcomp += '1'

# If sum = 0, No error is detected
if(int(finalcomp,2) == 0):
    print("Receiver Checksum is equal to 0. Therefore,")
    print("STATUS: ACCEPTED")
    
# Otherwise, Error is detected
else:
    print("Receiver Checksum is not equal to 0. Therefore,")
    print("STATUS: ERROR DETECTED")



class Agent:
    def __init__(self):
        pass

    def encode(self, message):
        return list(range(52))

    def decode(self, deck):
        return "NULL"
    