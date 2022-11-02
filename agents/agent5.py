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


class ASCII_Frequencies:
    """
    - Lowercase frequencies https://en.wikipedia.org/wiki/Letter_frequency
    - Uppercases frequencies https://link.springer.com/article/10.3758/BF03195586
    - Letters and numbers frequencies uses the above two sources and normalized the data
        * assumption for the above words, length of an average english word is 4.7
    - For all printable ascii values, we use https://github.com/piersy/ascii-char-frequency-english
    """
    lower_freq_with_space = {
        'a':	0.0703717,
        'b':	0.0115797,
        'c':	0.0262088,
        'd':	0.0316822,
        'e':	0.1035011,
        'f':	0.0173387,
        'g':	0.0161331,
        'h':	0.0395170,
        'i':	0.0605261,
        'j':	0.0008804,
        'k':	0.0061603,
        'l':	0.0341332,
        'm':	0.0196174,
        'n':	0.0606359,
        'o':	0.0632258,
        'p':	0.0167859,
        'q':	0.0007249,
        'r':	0.0553204,
        's':	0.0559656,
        't':	0.0736326,
        'u':	0.0215686,
        'v':	0.0087349,
        'w':	0.0135784,
        'x':	0.0016521,
        'y':	0.0141985,
        'z':	0.0008880,
        ' ':    0.1754386
    }

    upper_freq_with_space = {
        'A':	0.0656677,
        'B':	0.0396137,
        'C':	0.0536125,
        'D':	0.0303009,
        'E':	0.0323604,
        'F':	0.0235501,
        'G':	0.0217879,
        'H':	0.0288984,
        'I':	0.0521981,
        'J':	0.0183971,
        'K':	0.0108878,
        'L':	0.0250070,
        'M':	0.0606508,
        'N':	0.0480134,
        'O':	0.0247069,
        'P':	0.0337152,
        'Q':	0.0027252,
        'R':	0.0342315,
        'S':	0.0712855,
        'T':	0.0760752,
        'U':	0.0134375,
        'V':	0.0072585,
        'W':	0.0250563,
        'X':	0.0017713,
        'Y':	0.0220415,
        'Z':	0.0013113,
        ' ':    0.1754384
    }

    num_freq = {
        '0' : 1/10,
        '1' : 1/10,
        '2' : 1/10,
        '3' : 1/10,
        '4' : 1/10,
        '5' : 1/10,
        '6' : 1/10,
        '7' : 1/10,
        '8' : 1/10,
        '9' : 1/10
    }

    letter_freq_with_space = {
        ' ': 0.18125664573648903,
        'e': 0.09313797647646678,
        't': 0.06846864611028546,
        'a': 0.0662607652446656,
        'n': 0.05953428063624102,
        'i': 0.059284648793502086,
        'o': 0.05861850680392465,
        's': 0.05612630205699792,
        'r': 0.05573529672902756,
        'l': 0.034811609546727955,
        'd': 0.03449526751915381,
        'h': 0.028332631379469952,
        'c': 0.027045735008085783,
        'u': 0.020820570925370364,
        'm': 0.01962246149063737,
        'p': 0.018780802467988883,
        'f': 0.017037356126596872,
        'g': 0.013850967801605968,
        'y': 0.011783843369148178,
        'b': 0.01119188475057967,
        'w': 0.01034748327428945,
        'v': 0.008458069837816091,
        'k': 0.0053498414416146365,
        'S': 0.0033421596918414543,
        'T': 0.003320974237459051,
        'C': 0.0032315016873974453,
        'A': 0.0026799256987034845,
        'x': 0.0024948786392274126,
        'I': 0.0022619072023620204,
        'M': 0.001961677089933077,
        'B': 0.001880774707501893,
        'P': 0.0015025903503066305,
        'E': 0.001399542654718436,
        'N': 0.0013801397952031608,
        'F': 0.0013200114991081197,
        'R': 0.001193927192929351,
        'D': 0.0011820660809288472,
        'U': 0.0011278340601635368,
        'q': 0.0010909480586822002,
        'L': 0.0010865601328554242,
        'G': 0.0010070975385861514,
        'J': 0.0009534825698902312,
        'H': 0.0009467635584679804,
        'O': 0.0008881436118758939,
        'W': 0.0008705919085687896,
        'j': 0.0006680617071266566,
        'z': 0.0006233597127663754,
        'K': 0.00041191653698860356,
        'V': 0.0002765078884279358,
        'Y': 0.00027253133064742,
        'Q': 0.00010818979616644747,
        'Z': 9.324342381899147e-05,
        'X': 7.1098110661981e-05
    }

    all_printable_ascii = {
        ' ': 0.167564443682168, 
        'e': 0.08610229517681191, 
        't': 0.0632964962389326, 
        'a': 0.0612553996079051, 
        'n': 0.05503703643138501, 
        'i': 0.05480626188138746, 
        'o': 0.0541904405334676, 
        's': 0.0518864979648296, 
        'r': 0.051525029341199825, 
        'l': 0.03218192615049607, 
        'd': 0.03188948073064199, 
        'h': 0.02619237267611581, 
        'c': 0.02500268898936656, 
        '\n': 0.019578060965172565, 
        'u': 0.019247776378510318, 
        'm': 0.018140172626462205, 
        'p': 0.017362092874808832, 
        'f': 0.015750347191785568, 
        'g': 0.012804659959943725, 
        '.': 0.011055184780313847, 
        'y': 0.010893686962847832, 
        'b': 0.01034644514338097, 
        'w': 0.009565830104169261, 
        ',': 0.008634492219614468, 
        'v': 0.007819143740853554, 
        '0': 0.005918945715880591, 
        'k': 0.004945712204424292, 
        '1': 0.004937789430804492, 
        'S': 0.0030896915651553373, 
        'T': 0.0030701064687671904, 
        'C': 0.002987392712176473, 
        '2': 0.002756237869045172, 
        '8': 0.002552781042488694, 
        '5': 0.0025269211093936652,
        'A': 0.0024774830020061096, 
        '9': 0.002442242504945237, 
        'x': 0.0023064144740073764, 
        '3': 0.0021865587546870337, 
        'I': 0.0020910417959267183, 
        '-': 0.002076717421222119, 
        '6': 0.0019199098857390264, 
        '4': 0.0018385271551164353, 
        '7': 0.0018243295447897528, 
        'M': 0.0018134911904778657, 
        'B': 0.0017387002075069484, 
        '"': 0.0015754276887500987, 
        "'": 0.0015078622753204398, 
        'P': 0.00138908405321239, 
        'E': 0.0012938206232079082, 
        'N': 0.0012758834637326799, 
        'F': 0.001220297284016159, 
        'R': 0.0011037374385216535, 
        'D': 0.0010927723198318497, 
        'U': 0.0010426370083657518, 
        'q': 0.00100853739070613, 
        'L': 0.0010044809306127922, 
        'G': 0.0009310209736100016, 
        'J': 0.0008814561018445294, 
        'H': 0.0008752446473266058, 
        'O': 0.0008210528757671701, 
        'W': 0.0008048270353938186, 
        'j': 0.000617596049210692, 
        'z': 0.0005762708620098124, 
        '/': 0.000519607185080999, 
        '<': 0.00044107665296153596, 
        '>': 0.0004404428310719519, 
        'K': 0.0003808001912620934, 
        ')': 0.0003314254660634964, 
        '(': 0.0003307916441739124, 
        'V': 0.0002556203680692448, 
        'Y': 0.00025194420110965734, 
        ':': 0.00012036277683200988, 
        'Q': 0.00010001709417636208, 
        'Z': 8.619977698342993e-05, 
        'X': 6.572732994986532e-05, 
        ';': 7.41571610813331e-06, 
        '?': 4.626899793963519e-06, 
        '\x7f': 3.1057272589618137e-06, 
        '^': 2.2183766135441526e-06, 
        '&': 2.0282300466689395e-06, 
        '+': 1.5211725350017046e-06, 
        '[': 6.97204078542448e-07, 
        ']': 6.338218895840436e-07, 
        '$': 5.070575116672349e-07, 
        '!': 5.070575116672349e-07, 
        '*': 4.436753227088305e-07, 
        '=': 2.5352875583361743e-07, 
        '~': 1.9014656687521307e-07, 
        '_': 1.2676437791680872e-07, 
        '\t': 1.2676437791680872e-07, 
        '{': 6.338218895840436e-08, 
        '@': 6.338218895840436e-08, 
        '\x05': 6.338218895840436e-08, 
        '\x1b': 6.338218895840436e-08, 
        '\x1e': 6.338218895840436e-08
    }

encodings = {
    'lower': ASCII_Frequencies.lower_freq_with_space,
    'upper': ASCII_Frequencies.upper_freq_with_space,
    'printable': ASCII_Frequencies.all_printable_ascii,
    'number': ASCII_Frequencies.num_freq,
    'letters': ASCII_Frequencies.letter_freq_with_space
}

def generate_hoffman_code(type):
    _freq = sorted(encodings[type].items(), key = lambda x : x[1], reverse=True)
    node = make_tree(_freq)
    return huffman_code(node)

# HOFFMAN ENCODING FOR VARIETY OF WORDS
LOWERCASE_HOFFMAN = generate_hoffman_code('lower')
UPPERCASE_HOFFMAN = generate_hoffman_code('upper')
PRINTTABLE_HOFFMAN = generate_hoffman_code('printable')
NUMBER_HOFFMAN = generate_hoffman_code('number')
LETTERS_HOFFMAN = generate_hoffman_code('letters')

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
       
        _freq = sorted(ASCII_Frequencies.lower_freq_with_space.items(), key = lambda x : x[1], reverse=True)
        node = make_tree(_freq)
        self.encoding = huffman_code(node)
        
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
    agent = Agent()
    encoded = agent.encode('abcd')
    #print("ENCODED: ", encoded)
    decoded = agent.decode(encoded)
    print('Encoded msg: ', encoded)
    print('Decoded msg: ', decoded)
    
    