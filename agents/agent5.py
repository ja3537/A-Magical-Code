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

    # Domains definitions:
    G1_lower_num_punc = {
        ' ': 0.1780878364077455,
        'e': 0.09150969693107569,
        't': 0.06727164677467966,
        'a': 0.06510236506472278,
        'n': 0.05849347585309161,
        'i': 0.05824820818529814,
        'o': 0.05759371198637441,
        's': 0.05514507707171018,
        'r': 0.054760907472856536,
        'l': 0.034203017499622436,
        'd': 0.033892205904126955,
        'h': 0.027837307711428345,
        'c': 0.026572909434997873,
        'u': 0.02045657644058884,
        'm': 0.019279412888078322,
        'p': 0.018452468122949903,
        'f': 0.016739501486223718,
        'g': 0.013608819019675708,
        '.': 0.011749473189839063,
        'y': 0.011577832976288458,
        'b': 0.010996223241674676,
        'w': 0.010166583967699306,
        ',': 0.009176756142772862,
        'v': 0.008310202097594303,
        '0': 0.006290667716806582,
        'k': 0.005256313133184318,
        '1': 0.005247892793037605,
        '2': 0.002929335292959801,
        '8': 0.002713100957992203,
        '5': 0.002685616967753331,
        '9': 0.002595620372265259,
        'x': 0.002451262060790006,
        '3': 0.002323879155050527,
        '6': 0.0020404841870727435,
        '4': 0.0019539904530857044,
        '7': 0.0019389012035427944,
        'q': 0.0010718756193160179,
        'j': 0.0006563823551165961,
        'z': 0.0006124618609113395
    }

    G2_airport = {
        ' ': 0.7329604369403089,
        '0': 0.025890654024231335,
        '1': 0.021598879924588983,
        'S': 0.013514929718040422,
        'T': 0.013429260583880895,
        'C': 0.013067454046411046,
        '2': 0.012056336466217527,
        '8': 0.011166375557958358,
        '5': 0.011053259031301119,
        'A': 0.010837006847985805,
        '9': 0.01068285785577643,
        '3': 0.009564445923091853,
        'I': 0.009146635615071114,
        '6': 0.008398070365133493,
        '4': 0.008042086001829827,
        '7': 0.00797998281072389,
        'M': 0.007932573678227842,
        'B': 0.007605422939366215,
        'P': 0.006076131858382545,
        'E': 0.00565943053591727,
        'N': 0.005580969807868253,
        'F': 0.005337824724832959,
        'R': 0.00482796861570878,
        'D': 0.004780004990435,
        'U': 0.004560703096842164,
        'L': 0.004393800770744961,
        'G': 0.004072472206049516,
        'J': 0.0038556655299564722,
        'H': 0.0038284953838476255,
        'O': 0.0035914497213673795,
        'W': 0.0035204746458177383,
        'K': 0.0016656963043056364,
        'V': 0.001118134686295711,
        'Y': 0.0011020543957414954,
        'Q': 0.00043749480163020874,
        'Z': 0.0003770550888574676,
        'X': 0.0002875045052538191
    }

    G3_password = {
        'e': 0.11424631235218419,
        't': 0.08398604549695467,
        'a': 0.0812777813006228,
        'n': 0.07302683908909208,
        'i': 0.07272063190532521,
        'o': 0.0719035187502912,
        's': 0.06884649289752395,
        'r': 0.0683668720326455,
        'l': 0.042701142629641024,
        'd': 0.04231310639072306,
        'h': 0.034753800509678114,
        'c': 0.03317524823302224,
        'u': 0.02553924338147595,
        'm': 0.024069600278942047,
        'p': 0.023037191768114793,
        'f': 0.02089861926715143,
        'g': 0.016990083462272265,
        'y': 0.014454476049316765,
        'b': 0.013728358822004733,
        'w': 0.01269258632124321,
        'v': 0.010374965456028363,
        '0': 0.007853655000293508,
        'k': 0.006562303364275874,
        '1': 0.006551790895731282,
        '2': 0.00365716165704089,
        '8': 0.003387201464815777,
        '5': 0.00335288876748623,
        '9': 0.0032405315036816346,
        'x': 0.003060305742953156,
        '3': 0.002901273118810574,
        '6': 0.0025474654774737984,
        '4': 0.002439481400583753,
        '7': 0.0024206430569518452,
        'q': 0.001338195195852335,
        'j': 0.000819467947988006,
        'z': 0.0007646349120594161,
        '@': 8.409974835673296e-08
    }

    G4_location = {
        '0' : 1/17,
        '1' : 1/17,
        '2' : 1/17,
        '3' : 1/17,
        '4' : 1/17,
        '5' : 1/17,
        '6' : 1/17,
        '7' : 1/17,
        '8' : 1/17,
        '9' : 1/17,
        '.' : 2/17,
        ' ' : 3/17,
        ',' : 1/17,
        'N' : 1/17,
        'S' : 1/17,
        'E' : 1/17,
        'W' : 1/17
    }

encodings = {
    'lower': ASCII_Frequencies.G1_lower_num_punc,
    'airport': ASCII_Frequencies.G2_airport,
    'password': ASCII_Frequencies.G3_password,
    'location': ASCII_Frequencies.G4_location,
    # 'lower': ASCII_Frequencies.lower_freq_with_space,
    # 'upper': ASCII_Frequencies.upper_freq_with_space,
    'printable': ASCII_Frequencies.all_printable_ascii,
    # 'number': ASCII_Frequencies.num_freq,
    # 'letters': ASCII_Frequencies.letter_freq_with_space
}

def generate_huffman_code(type):
    _freq = sorted(encodings[type].items(), key = lambda x : x[1], reverse=True)
    node = make_tree(_freq)
    return huffman_code(node)

# HUFFMAN ENCODING FOR VARIETY OF WORDS
LOWERCASE_HUFFMAN = generate_huffman_code('lower')
AIRPORT_HUFFMAN = generate_huffman_code('airport')
PASSPORT_HUFFMAN = generate_huffman_code('password')
LOCATION_HUFFMAN = generate_huffman_code('location')
PRINTTABLE_HUFFMAN = generate_huffman_code('printable')
# NUMBER_HUFFMAN = generate_huffman_code('number')
# LETTERS_HUFFMAN = generate_huffman_code('letters')

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

    # print("permutation is ", permutations)
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
       self.scheme_id_to_encoding = { "001" : LOWERCASE_HUFFMAN,
                                      "010" : AIRPORT_HUFFMAN,
                                      "011" : PASSPORT_HUFFMAN,
                                      "100" : LOCATION_HUFFMAN,
                                    #   "011" : LETTERS_HUFFMAN,
                                    #   "100" : NUMBER_HUFFMAN,
                                      "101" : PRINTTABLE_HUFFMAN
                                    }

       self.encoding_to_scheme_id = { "LOWERCASE_HUFFMAN" : "001",
                                      "AIRPORT_HUFFMAN" : "010",
                                      "PASSPORT_HUFFMAN" : "011",
                                      "LOCATION_HUFFMAN" : "100",
                                    #   "LETTERS_HUFFMAN" : "011",
                                    #   "NUMBER_HUFFMAN" : "100",
                                      "PRINTTABLE_HUFFMAN" : "101"
                                    }
        #self.encoding = huffman_code(node)
        
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
        ms = set(message)
        if ms.issubset(ASCII_Frequencies.G1_lower_num_punc.keys()):
            encoding = LOWERCASE_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["LOWERCASE_HUFFMAN"]
        elif ms.issubset(ASCII_Frequencies.G2_airport.keys()):
            encoding = AIRPORT_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["AIRPORT_HUFFMAN"]
        elif ms.issubset(ASCII_Frequencies.G3_password.keys()):
            encoding = PASSPORT_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["PASSPORT_HUFFMAN"]
        elif ms.issubset(ASCII_Frequencies.G4_location.keys()):
            encoding = LOCATION_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["LOCATION_HUFFMAN"]
        # elif ms.issubset(ASCII_Frequencies.letter_freq_with_space.keys()):
        #     encoding = LETTERS_HUFFMAN
        #     scheme_id = self.encoding_to_scheme_id["LETTERS_HUFFMAN"]
        # elif ms.issubset(ASCII_Frequencies.num_freq.keys()):
        #     encoding = NUMBER_HUFFMAN
        #     scheme_id = self.encoding_to_scheme_id["NUMBER_HUFFMAN"]
        # else:
        #     encoding = PRINTTABLE_HUFFMAN
        #     scheme_id = self.encoding_to_scheme_id["PRINTTABLE_HUFFMAN"]
        
        msg_huffman_binary = encode_msg_bin(message, encoding)
       
        #print("encoded scheme id is ", scheme_id)
        
        
        # Calculate checksum before prepending the leading 1 bit
        # assert(len(self.compute_crc16_checksum(msg_huffman_binary)) == 16)
        msg_huffman_binary += self.compute_crc16_checksum(msg_huffman_binary)

        # Appending 3-bit identifier for encoding scheme
        msg_huffman_binary += scheme_id
        
        msg_huffman_binary = "1" + msg_huffman_binary
        
        cards = bin_to_cards(msg_huffman_binary)
        
        return cards



    def decode(self, deck):
        """
        Given a binary str, use 'decode_bin_msg' to decode it
        see main below
        """
        #print("after shuffling ", deck)
      
        for perm_bound in range(1, 52):
            msg_cards = []
            for c in deck:
                if c <= perm_bound:
                    msg_cards.append(c)
            bin_raw = cards_to_bin(msg_cards)
            bin_raw = bin_raw[1:] # remove leading 1
            bin_message, tail = bin_raw[:-19], bin_raw[-19:]
            checksum, scheme_id = tail[:-3], tail[-3:]

            if scheme_id in self.scheme_id_to_encoding and checksum == self.compute_crc16_checksum(bin_message):
               #print("scheme_id ", scheme_id)
               decoded_message = decode_bin_msg(bin_message, self.scheme_id_to_encoding[scheme_id])
               return decoded_message
        return "NULL"


if __name__=='__main__':
    agent = Agent()
    encoded = agent.encode('abcd')
    #print("ENCODED: ", encoded)
    decoded = agent.decode(encoded)
    print('Encoded msg: ', encoded)
    print('Decoded msg: ', decoded)
    
    