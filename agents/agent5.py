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
        

    def encode(self, message):
        """
        FYI: use 'encode_msg_bin' to compress a message to binary
        """
        msg_huffman_binary = encode_msg_bin(message, self.encoding)
        
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
        return msg_huffman_binary


    def decode(self, deck):
        """
        Given a binary str, use 'decode_bin_msg' to decode it
        see main below
        """

        # msg = ''

        # in_msg = False
        # for x in deck:
        #     if x == self.encoder['#']:
        #         in_msg = True
        #     elif x == self.encoder['/']:
        #         in_msg = False
        #     elif in_msg and x in self.decoder:
        #         msg += self.decoder[x]

        # return msg

if __name__=='__main__':
    _freq = sorted(freq.items(), key = lambda x : x[1], reverse=True)
    node = make_tree(_freq)
    encoding = huffman_code(node)
    
    obj = Agent()
    encoded = obj.encode('rwras')
    decoded = decode_bin_msg(encoded, encoding)
    print('Encoded msg: ', encoded)
    print('Decoded msg: ', decoded)