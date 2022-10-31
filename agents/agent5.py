from collections import Counter
import numpy as np
import math
import random

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

def encode_msg_bin(msg, encoding) -> list[int]:
    """
    takes a string and returns a binary encoding of each letter per 'encoding'
    """
    binaries = []
    for letter in msg:
        binaries.append(encoding[letter])
    return "".join(binaries)

def decode_bin_msg(msg, encoding):
    """
    takes a binary str and decodes the message according to 'encoding'
    """
    output = ''
    while msg:
        for ch, binary in encoding.items():
            if msg.startswith(binary):
                output += ch
                msg = msg[len(binary) :]
            
    return output

def encode_bin_to_cards(msg_bin):
    """
    takes a binary string and encodes the string into cards
    """
    digit = int(msg_bin, 2)
    #digit = 16
    m=digit

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
    for i in range(min_cards, 51):
        remaining_cards.append(i)

    random.shuffle(remaining_cards)

    remaining_cards.append(51)

    returned_list = remaining_cards + permutations

   # print(permutations)
   # print(returned_list)


    return returned_list

def decode_cards_to_bin(cards):
    """
    takes a binary string and encodes the string into cards
    """
    '''
    fact_list = []
    while(len(cards)>0):
        min_index = np.argmin(cards)
        min_value = cards[min_index]
        num_digits = 0
        for i in range(min_index):
            current_card = cards[i]
            if current_card > min_value:
                num_digits += 1
        fact_list.append(num_digits)
        cards.remove(min_value)

    fact_list.reverse()

    digit = 0
    for i in range(len(fact_list)):
        fact  = math.factorial(i)
        product = fact*fact_list[i]
        digit += product

    binary = bin(digit)
    '''
    m = 1
    digit = 0
    length = len(cards)
    positions = []
    elements = []
    for i in range(length):
        positions.append(i)
        elements.append(i)

    #print(positions)
   # print(elements)
    for i in range(length-1):
        digit += m*positions[cards[i]]
        m = m * (length - i)
        positions[elements[length-i-1]] = positions[cards[i]]
        elements[positions[cards[i]]] = elements[length-i-1]
      #  print(digit)
       # print(m)
       # print(positions)
       # print(elements)



    return digit

# def str_to_bin(self, msg) -> str:
#     """
#     str_to_bin transforms string from ascii character to binary
#     returns a list of bytes
#     """
#     encoded_bin = []
#     for ch in msg:
#         # char to ascii key
#         ascii_rep = ord(ch)
#         # ascii to binary
#         binary_rep = bin(ascii_rep)[2:]
#         # padding = ['0'] * (8 - len(binary_rep)) if len(binary_rep) < 8 else []
#         encoded_bin.append(binary_rep.zfill(8))

#     frequencies = dict(Counter(msg))
#     frequencies = sorted(frequencies.items(), key = lambda x : x[1], reverse=True)
#     node = make_tree(frequencies)
#     encoding = huffman_code(node)

#     print('Fre: ', frequencies, '\n\n')
#     print('Encoding:', encoding)

#     return encoded_bin

# def bin_to_str(self, bytes) -> str:
#     """
#     bin_to_str transforms a list of bytes to string
#     returns a string
#     """
#     decoded_str = ''
#     for byte in bytes:
#         # converts binary to decimal then maps decimal to ascii
#         ch = chr(int(byte, 2))
#         decoded_str += ch
#     return decoded_str
class Agent:

    def __init__(self):
       
        _freq = sorted(freq.items(), key = lambda x : x[1], reverse=True)
        node = make_tree(_freq)
        self.encoding = huffman_code(node)
        self.start_card = 51
        
        
        # self.encoder = {}
        # self.decoder = {}
        # # 1 start of msg, 26 letters of the alphabet, 10 numbers, and 1 end msg signal

        # for i in range(26):
        #     self.encoder[chr(i + ord('a'))] = i
        #     self.decoder[i] = chr(i + ord('a'))

        # for i in range(10):
        #     self.encoder[str(i)] = i + 26
        #     self.decoder[i + 26] = str(i)

        # self.encoder['#'] = 36
        # self.decoder[36] = '#'
        # self.encoder['/'] = 37
        # self.decoder[37] = '/'
        
    def compute_crc_checksum(self, data) -> str:
        # data is a binary string
        # we would like to turn it into a list of bytes
        # then compute the crc and return the crc as a binary string
        if len(data) % 8 != 0:
            data = "0" * (8 - len(data) % 8) + data
        
        byte_list = [int(data[i:i+8], 2) for i in range(0, len(data), 8)]
        print(byte_list)
        generator = 0x1D
        crc = 0
        for currByte in byte_list:
            crc ^= currByte
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


    def encode(self, message):
        """
        FYI: use 'encode_msg_bin' to compress a message to binary
        """
        msg_huffman_binary = encode_msg_bin(message, self.encoding)
        #print(type(msg_huffman_binary))
        msg_huffman_binary = str(1) + msg_huffman_binary
        #print(msg_huffman_binary)
        #print(int(msg_huffman_binary, 2))
        cards = encode_bin_to_cards(msg_huffman_binary)
        
        # print('msg passed to encoder bytes: ', self.str_to_bin(message))
        # print('msg in str from decoded bytes: ', self.bin_to_str(self.str_to_bin(message)))
        
        # msgList = []

        # # start '#', end '/'
        # message = '#' + message + '/'
        
        # encoded = list(range(38, 52)) # red section
        # availL = set(range(38))
        
        # for x in message:
        #     msgList.append(self.encode[x])
        #     availL.remove(self.encoder[x])

        # encoded += list(availL)
        # encoded += msgList

        # return encoded
       # list_52 = np.arange(52)
       # list_52 = list(list_52)
       # list_52[41] = 51
       # list_52[51] = 41
        #print(list_52)
        print(cards)
        return cards



    def decode(self, deck):
        """
        Given a binary str, use 'decode_bin_msg' to decode it
        see main below
        """
        #print(deck)
     #   list_52 = np.arange(52)
      #  list_52 = list(list_52)
       # list_52.reverse()
       # list_52[41] = 51
        #list_52[0] = 10
        #deck = list_52
        print(deck)

        message_length = deck.index(self.start_card)

        bottom_cards = []
        for i in range(len(deck)):
            if i > message_length:
                bottom_cards.append(deck[i])

        print(bottom_cards)

        binary_message = decode_cards_to_bin(bottom_cards)
        print(binary_message)
        binary_message = bin(int(binary_message))[3:]
        print(binary_message)

        decoded_message = decode_bin_msg(binary_message, self.encoding)

        print(binary_message)

        # msg = ''

        # in_msg = False
        # for x in deck:
        #     if x == self.encoder['#']:
        #         in_msg = True
        #     elif x == self.encoder['/']:
        #         in_msg = False
        #     elif in_msg and x in self.decoder:
        #         msg += self.decoder[x]

        return decoded_message

if __name__=='__main__':
    _freq = sorted(freq.items(), key = lambda x : x[1], reverse=True)
    node = make_tree(_freq)
    encoding = huffman_code(node)
    
    obj = Agent()
    encoded = obj.encode('0000')
    decoded = decode_bin_msg(encoded, encoding)
    print('Encoded msg: ', encoded)
    print('Decoded msg: ', decoded)