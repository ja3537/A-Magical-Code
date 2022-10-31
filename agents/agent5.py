import math
import random
from collections import Counter

class Node:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right
    
def huffman_code(node, bin_str=''):
    if type(node) is str:
        return {node : bin_str}
    l, r = node.children()
    _dict = dict()
    _dict.update(huffman_code(l, bin_str + '0'))
    _dict.update(huffman_code(r, bin_str + '1'))
    return _dict

def make_tree(nodes):
    """
    make tree buils the huffman tree and returns the root of the tree
    """
    while len(nodes) > 1:
        
        k1, v1 = nodes[-1]
        k2, v2 = nodes[-2]
        nodes = nodes[:-2] # saves the whole
        combined_node = Node(k1, k2)
        nodes.append((combined_node, v1 + v2))
        nodes = sorted(nodes, key=lambda x : x[1], reverse=True)
    return nodes[0][0] # root



# Relative frequencies of letters https://en.wikipedia.org/wiki/Letter_frequency
freq = {
    'a' :   8.167/100,
    'b' :   1.492/100,	
    'c' :   2.782/100,	
    'd'	:   4.253/100,	
    'e'	:   12.702/100,
    'f' :   2.228/100,
    'g' :   2.015/100,
    'h' : 	6.094/100,
    'i' :	6.966/100,
    'j' :	0.153/100,
    'k' : 	0.772/100,
    'l' :	4.025/100,
    'm' :	2.406/100,
    'n' :	6.749/100,
    'o' :	7.507/100,
    'p' :	1.929/100,
    'q' :	0.095/100,
    'r' :	5.987/100,
    's' :	6.327/100,
    't' :	9.056/100,
    'u' :	2.758/100,
    'v' :	0.978/100,
    'w' :	2.360/100,
    'x' :	0.150/100,
    'y' :	1.974/100,
    'z' :	0.074/100
}

def encode_msg_bin(msg, encoding) -> str:
    """
    takes a string and returns a binary encoding of each letter per 'encoding'
    """
    binaries = []
    for letter in msg:
        binaries.append(encoding[letter])
    return "".join(binaries)

def decode_bin_msg(msg, encoding) -> str:
    """
    takes a binary str and decodes the message according to 'encoding'
    """
    output = ''
    while msg:
        found_match = False
        for ch, binary in encoding.items():
            if msg.startswith(binary):
                found_match = True
                output += ch
                msg = msg[len(binary):]
        if not found_match:
            break # break and returns partial msg
    return output

def bin_to_cards(msg_bin):
    """
    takes a binary string and encodes the string into cards
    """
    digit = int(msg_bin, 2)
    #digit = 16
    m = digit

    min_cards = math.inf
    for i in range(1, 53):
        fact = math.factorial(i) - 1
        if digit < fact:
            min_cards = i
            break
    #print(min_cards)
    permutations = []
    elements = []
    for i in range(min_cards):
        elements.append(i)
        permutations.append(0)
    for i in range(min_cards):
        index = m % (min_cards-i)
        #print(index)
        m = m // (min_cards-i)
        permutations[i] = elements[index]
        elements[index] = elements[min_cards-i-1]

    remaining_cards = []
    for i in range(min_cards, 52):
        remaining_cards.append(i)

    random.shuffle(remaining_cards)

    print("permutation is ", permutations)
    returned_list = remaining_cards + permutations

   # print(permutations)
   # print(returned_list)


    return returned_list

def cards_to_bin(cards):
    """
    takes a binary string and encodes the string into cards
    """
    m = 1
    digit = 0
    length = len(cards)
    positions = []
    elements = []
    for i in range(length):
        positions.append(i)
        elements.append(i)

    for i in range(length-1):
        digit += m * positions[cards[i]]
        m = m * (length - i)
        positions[elements[length-i-1]] = positions[cards[i]]
        elements[positions[cards[i]]] = elements[length-i-1]
     
    return format(digit, 'b')

class Agent:

    def __init__(self):
       
        _freq = sorted(freq.items(), key = lambda x : x[1], reverse=True)
        node = make_tree(_freq)
        self.encoding = huffman_code(node)
        # self.start_card = 51
        
    def compute_crc8_checksum(self, data) -> str:
        # data is a binary string
        # we would like to turn it into a list of bytes
        # then compute the crc and return the crc as a binary string
        if len(data) % 8 != 0:
            data = "0" * (8 - len(data) % 8) + data
        
        byte_list = [int(data[i:i+8], 2) for i in range(0, len(data), 8)]
        generator = 0x9B
        crc = 0
        for curr_byte in byte_list:
            crc ^= curr_byte
            # mask to trim to 8 bits
            crc &= 0xFF
            for i in range(8):
                if crc & 0x80 != 0:
                    crc = (crc << 1) ^ generator
                    # mask to trim to 8 bits
                    crc &= 0xFF
                else:
                    crc = crc << 1
        return format(crc, '08b')

    def compute_crc16_checksum(self, data) -> str:
        if len(data) % 8 != 0:
            data = "0" * (8 - len(data) % 8) + data
        
        byte_list = [int(data[i:i+8], 2) for i in range(0, len(data), 8)]
        # polynomial generator picked based on https://users.ece.cmu.edu/~koopman/crc/
        generator = 0xED2F
        crc = 0
        for curr_byte in byte_list:
            crc ^= (curr_byte << 8)
            # mask to trim to 16 bits
            crc &= 0xFFFF
            for i in range(8):
                if crc & 0x8000 != 0:
                    crc = (crc << 1) ^ generator
                    # mask to trim to 16 bits
                    crc &= 0xFFFF
                else:
                    crc = crc << 1
        return format(crc, '016b')

        

    def encode(self, message):
        """
        FYI: use 'encode_msg_bin' to compress a message to binary
        """
        msg_huffman_binary = encode_msg_bin(message, self.encoding)
       
        # Calculate checksum before prepending the leading 1 bit
        # assert(len(self.compute_crc16_checksum(msg_huffman_binary)) == 16)
        msg_huffman_binary += self.compute_crc16_checksum(msg_huffman_binary)
        msg_huffman_binary = "1" + msg_huffman_binary
        
        cards = bin_to_cards(msg_huffman_binary)
        
        return cards



    def decode(self, deck):
        """
        Given a binary str, use 'decode_bin_msg' to decode it
        see main below
        """
        print("after shuffling ", deck)
      
        for perm_bound in range(1, 52):
            msg_cards = []
            for c in deck:
                if c <= perm_bound:
                    msg_cards.append(c)
            bin_raw = cards_to_bin(msg_cards)
            bin_raw = bin_raw[1:] # remove leading 1
            bin_message, checksum = bin_raw[:-16], bin_raw[-16:]
            if checksum == self.compute_crc16_checksum(bin_message):
               decoded_message = decode_bin_msg(bin_message, self.encoding)
               return decoded_message
        return "NULL"


if __name__=='__main__':
    _freq = sorted(freq.items(), key = lambda x : x[1], reverse=True)
    node = make_tree(_freq)
    encoding = huffman_code(node)
    
    agent = Agent()
    encoded = agent.encode('abcd')
    #print("ENCODED: ", encoded)
    decoded = agent.decode(encoded)
    print('Encoded msg: ', encoded)
    print('Decoded msg: ', decoded)