import math
import random
from collections import Counter
import enchant


class Vocab:
    def __init__(self, vocab_list, pad_symbol):
        self.pad_symbol = pad_symbol
        self.word_to_index = dict()
        self.index_to_word = dict()
        self.word_to_index[self.pad_symbol] = 0
        self.index_to_word[0] = self.pad_symbol
        for i, word in enumerate(vocab_list, start=1):
            self.word_to_index[word] = i
            self.index_to_word[i] = word


class DictEncoding:
    # Dictionary Encoding for Group 6, 7, 8
    def __init__(self, g6_vocab_list, g7_vocab_list, g8_vocab_list):
        self.max_len = 6
        self.pad_symbol = ''
        # use cards from 0 to 27 (28 cards) to encode message + 3-bit encoding scheme
        self.num_cards_used = 28
        self.factorial_table = [math.factorial(
            self.num_cards_used-i-1) for i in range(self.num_cards_used)]
        self.group_vocab = dict()

        # Group 6
        self.group_vocab[6] = Vocab(g6_vocab_list, self.pad_symbol)

        # Group 7
        self.group_vocab[7] = Vocab(g7_vocab_list, self.pad_symbol)

        # Group 8
        self.group_vocab[8] = Vocab(g8_vocab_list, self.pad_symbol)

    # Based on Group 7's implementation

    def nth_perm(self, n):
        perm = []
        items = list(range(self.num_cards_used))
        for f in self.factorial_table:
            lehmer = n // f
            perm.append(items.pop(lehmer))
            n %= f
        return perm

    def perm_number(self, permutation):
        n = len(permutation)
        # if n != 28:
        #     print(permutation)
        number = 0
        for i in range(n):
            k = 0
            for j in range(i + 1, n):
                if permutation[j] < permutation[i]:
                    k += 1
            number += k * self.factorial_table[i]
        return number

    def encode(self, message, group_id):
        """Return message_section, encoding_scheme, truncated"""
        # if group_id != 6:
        #     print(message)
        tokens = message.split()
        if len(tokens) > self.max_len:
            truncated = "1"
        else:
            truncated = "0"
        padded_tokens = tokens + [self.pad_symbol] * (self.max_len - len(tokens))
        word_to_index = self.group_vocab[group_id].word_to_index
        num = 0
        for i in range(self.max_len):
            num += word_to_index.get(padded_tokens[i], 0) * \
                                     (len(word_to_index) ** i)
        num = format(num, 'b')
        encoding_scheme = format(group_id - 1, "03b")
        # return num, encoding_scheme, truncated
        full_message = num + encoding_scheme + truncated
        num = int(full_message, 2)
        deck = list(range(self.num_cards_used, 50)) + self.nth_perm(num) + [50, 51]
        # print("Encoding group {} deck: {}".format(group_id, deck))
        return deck
        # return bin_to_cards(full_message)

    def decode(self, cards):
        # Pass in only cards from 0 to 27
        num = self.perm_number(cards)
        num = format(num, 'b')
        num, scheme_id, truncated = num[:-4], num[-4:-1], num[-1]
        # if num == "" or scheme_id == "" or truncated == "":
        #     return "NULL"
        group_id = int(scheme_id, 2) + 1
 
        word_to_index = self.group_vocab[group_id].word_to_index
        index_to_word = self.group_vocab[group_id].index_to_word
        num = int(num, 2)
        words = []
        for i in range(self.max_len):
            index = num % len(word_to_index)
            words.append(index_to_word[index])
            num = num // len(word_to_index)
        result = ' '.join(words).rstrip()
        if truncated == "1":
            result = "PARTIAL: " + result
        return result


class Node:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right


def huffman_code(node, bin_str=''):
    if type(node) is str:
        return {node: bin_str}
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
        nodes = nodes[:-2]  # saves the whole
        combined_node = Node(k1, k2)
        nodes.append((combined_node, v1 + v2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]  # root


class ASCII_Frequencies:
    """
    - Lowercase frequencies https://en.wikipedia.org/wiki/Letter_frequency
    - Uppercases frequencies https://link.springer.com/article/10.3758/BF03195586
    - Letters and numbers frequencies uses the above two sources and normalized the data
        * assumption for the above words, length of an average english word is 4.7
    - For all printable ascii values, we use https://github.com/piersy/ascii-char-frequency-english
    """

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
        '0': 1/17,
        '1': 1/17,
        '2': 1/17,
        '3': 1/17,
        '4': 1/17,
        '5': 1/17,
        '6': 1/17,
        '7': 1/17,
        '8': 1/17,
        '9': 1/17,
        '.': 2/17,
        ' ': 3/17,
        ',': 1/17,
        'N': 1/17,
        'S': 1/17,
        'E': 1/17,
        'W': 1/17
    }

    G5_addresses = {
        ' ': 0.1718380167599127,
        'e': 0.08829825299765842,
        't': 0.06491081366987489,
        'a': 0.0628176607945765,
        'n': 0.05644070413082936,
        'i': 0.056204043893616634,
        'o': 0.05557251660310336,
        's': 0.053209814152490174,
        'r': 0.05283912661257692,
        'l': 0.03300269582074661,
        'd': 0.03270279185631389,
        'h': 0.02686038443476325,
        'c': 0.025640358987775367,
        'u': 0.019738672759210717,
        'm': 0.01860282061823506,
        'p': 0.017804896676454386,
        'f': 0.01615204494009558,
        'g': 0.013131230733981479,
        '.': 0.011337136840121092,
        'y': 0.011171520173156514,
        'b': 0.01061032146728047,
        'w': 0.009809797577823746,
        ',': 0.008854706799025669,
        'v': 0.008018563626407018,
        '0': 0.006069902843205279,
        'k': 0.0050718479290604275,
        '1': 0.005063723092572761,
        'S': 0.003168491234113913,
        'T': 0.0031484066383164035,
        'C': 0.003063583345385174,
        '2': 0.00282653311602104,
        '8': 0.002617887315017785,
        '5': 0.0025913678487220443,
        'A': 0.0025406688690390103,
        '9': 0.0025045295963418736,
        'x': 0.002365237399597333,
        '3': 0.0022423248732119263,
        'I': 0.002144371844516629,
        '-': 0.002129682140146929,
        '6': 0.00196887537638305,
        '4': 0.0018854170559817482,
        '7': 0.0018708573489958514,
        'M': 0.0018597425726807246,
        'B': 0.0017830441162371606,
        'P': 0.0014245113317094492,
        'E': 0.001326818297781757,
        'N': 0.001308423667973682,
        'F': 0.0012514198151762194,
        'R': 0.0011318872207696819,
        'D': 0.0011206424470707526,
        'U': 0.0010692284817768042,
        'q': 0.001034259185533891,
        'L': 0.001030099269252206,
        'G': 0.0009547657853385699,
        'J': 0.0009039368082717334,
        'H': 0.0008975669364654035,
        'O': 0.0008419930548897703,
        'W': 0.0008253533897630309,
        'j': 0.0006333472538865155,
        'z': 0.0005909681067668512,
        'K': 0.0003905121409431635,
        'V': 0.0002621397244380457,
        'Y': 0.0002583698003077688,
        'Q': 0.0001025679358202916,
        'Z': 8.839822098580264e-05,
        'X': 6.740364350167453e-05
    }


encodings = {
    'lower': ASCII_Frequencies.G1_lower_num_punc,
    'airport': ASCII_Frequencies.G2_airport,
    'password': ASCII_Frequencies.G3_password,
    'location': ASCII_Frequencies.G4_location,
    'address': ASCII_Frequencies.G5_addresses,
    # 'lower': ASCII_Frequencies.lower_freq_with_space,
    # 'upper': ASCII_Frequencies.upper_freq_with_space,
    'printable': ASCII_Frequencies.all_printable_ascii,
    # 'number': ASCII_Frequencies.num_freq,
    # 'letters': ASCII_Frequencies.letter_freq_with_space
}


def generate_huffman_code(type):
    _freq = sorted(encodings[type].items(), key=lambda x: x[1], reverse=True)
    node = make_tree(_freq)
    return huffman_code(node)


# HUFFMAN ENCODING FOR VARIETY OF WORDS
LOWERCASE_HUFFMAN = generate_huffman_code('lower')
AIRPORT_HUFFMAN = generate_huffman_code('airport')
PASSPORT_HUFFMAN = generate_huffman_code('password')
LOCATION_HUFFMAN = generate_huffman_code('location')
ADDRESS_HUFFMAN = generate_huffman_code('address')
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
            break  # break and returns partial msg
    return output


def bin_to_cards(msg_bin):
    """
    takes a binary string and encodes the string into cards
    """
    digit = int(msg_bin, 2)
    # digit = 16
    m = digit

    min_cards = math.inf
    for i in range(1, 53):
        fact = math.factorial(i) - 1
        if digit < fact:
            min_cards = i
            break
    # print(min_cards)
    permutations = []
    elements = []
    for i in range(min_cards):
        elements.append(i)
        permutations.append(0)
    for i in range(min_cards):
        index = m % (min_cards-i)
        # print(index)
        m = m // (min_cards-i)
        permutations[i] = elements[index]
        elements[index] = elements[min_cards-i-1]

    remaining_cards = []
    for i in range(min_cards, 52):
        remaining_cards.append(i)

    random.shuffle(remaining_cards)
    # print("Num cards used = ", len(permutations))

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
        self.pearson8_table = list(range(0, 256))
        random.shuffle(self.pearson8_table)
        self.english_dict = enchant.Dict("en_US")
        self.scheme_id_to_encoding = { "000" : LOWERCASE_HUFFMAN,
                                        "001" : AIRPORT_HUFFMAN,
                                        "010" : PASSPORT_HUFFMAN,
                                        "011" : LOCATION_HUFFMAN,
                                        "100" : ADDRESS_HUFFMAN,
                                        #   "011" : LETTERS_HUFFMAN,
                                        #   "100" : NUMBER_HUFFMAN,
                                        # "110" : PRINTTABLE_HUFFMAN
                                        }

        self.encoding_to_scheme_id = { "LOWERCASE_HUFFMAN" : "000",
                                        "AIRPORT_HUFFMAN" : "001",
                                        "PASSPORT_HUFFMAN" : "010",
                                        "LOCATION_HUFFMAN" : "011",
                                        "ADDRESS_HUFFMAN" : "100",
                                        #   "LETTERS_HUFFMAN" : "011",
                                        #   "NUMBER_HUFFMAN" : "100",
                                        #  "PRINTTABLE_HUFFMAN" : "110"
                                        }

        # Group 6
        self.g6_vocab_list = []
        for ng in range(1, 10):
            with open("messages/agent6/corpus-ngram-"+str(ng)+".txt", 'r') as f:
                line = f.readline()
                while line:
                    line = line.strip()
                    words = line.split(" ")
                    for word in words:
                        self.g6_vocab_list.append(word)
                    line = f.readline()
        # Group 7
        with open('messages/agent7/30k.txt', 'r') as f:
            self.g7_vocab_list = []
            line = f.readline()
            while line:
                line = line.strip()
                if self.english_dict.check(line):
                    self.g7_vocab_list.append(line)
                line = f.readline()

        # Group 8
        self.g8_vocab_list = []
        with open('messages/agent8/names.txt', 'r') as f:
            self.g8_vocab_list.extend(f.read().splitlines())
        with open('messages/agent8/places.txt', 'r') as f:
            self.g8_vocab_list.extend(f.read().splitlines())
        
        self.g6_vocab_set = set(self.g6_vocab_list)
        self.g7_vocab_set = set(self.g7_vocab_list)
        self.g8_vocab_set = set(self.g8_vocab_list)
        
        
        self.g6_vocab_list = list(self.g6_vocab_set)
        self.g7_vocab_list = list(self.g7_vocab_set)
        self.g8_vocab_list = list(self.g8_vocab_set)
        self.dict_encoding = DictEncoding(self.g6_vocab_list, self.g7_vocab_list, self.g8_vocab_list)
        

    
    def compute_pearson8_checksum(self, data) -> str:
        if len(data) % 8 != 0:
            data = "0" * (8 - len(data) % 8) + data
        byte_list = [int(data[i:i+8], 2) for i in range(0, len(data), 8)]
        checksum = len(data) % 256
        checksum &= 0xFF
        for byte in byte_list:
            index = (checksum ^ byte) & 0xFF
            checksum = self.pearson8_table[index]
            checksum &= 0xFF
        return format(checksum, '08b')

        
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

    
    def compose_huffman_bin(self, data, checksum, scheme_id, truncated):
        return "1" + data + checksum + scheme_id + truncated
    
    def decompose_huffman_bin(self, msg):
        msg = msg[1:]
        data = msg[:-12]
        checksum = msg[-12:-4]
        scheme_id = msg[-4:-1]
        truncated = msg[-1:]
        if data == "" or scheme_id == "" or truncated == "" or checksum == "":
            return None
        else:
            return data, scheme_id, truncated, checksum
    
    def compose_dict_bin(self, data, scheme_id, truncated):
        return "1" + data + scheme_id + truncated
    
    def decompose_dict_bin(self, msg):
        msg = msg[1:]
        data = msg[:-4]
        scheme_id = msg[-4:-1]
        truncated = msg[-1]
        # if data == "" or scheme_id == "" or truncated == "":
        #     return None
        # else:
        return data, scheme_id, truncated

    def encode(self, message):
        """
        FYI: use 'encode_msg_bin' to compress a message to binary
        """
        truncated = "0"
        ms = set(message)
        tokens = set(message.split())
        if tokens.issubset(self.g6_vocab_set):
            #msg_binary, scheme_id, truncated = self.dict_encoding.encode(message, 6)
            return self.dict_encoding.encode(message, 6)
        elif tokens.issubset(self.g7_vocab_set):
            #msg_binary, scheme_id, truncated = self.dict_encoding.encode(message, 7)
            return self.dict_encoding.encode(message, 7)
        elif tokens.issubset(self.g8_vocab_set):
            #msg_binary, scheme_id, truncated = self.dict_encoding.encode(message, 8)
            return self.dict_encoding.encode(message, 8)
        elif ms.issubset(ASCII_Frequencies.G1_lower_num_punc.keys()):
            encoding = LOWERCASE_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["LOWERCASE_HUFFMAN"]
            msg_binary = encode_msg_bin(message, encoding)
        elif ms.issubset(ASCII_Frequencies.G2_airport.keys()):
            encoding = AIRPORT_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["AIRPORT_HUFFMAN"]
            msg_binary = encode_msg_bin(message, encoding)
        elif ms.issubset(ASCII_Frequencies.G3_password.keys()):
            encoding = PASSPORT_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["PASSPORT_HUFFMAN"]
            msg_binary = encode_msg_bin(message, encoding)
        elif ms.issubset(ASCII_Frequencies.G4_location.keys()):
            encoding = LOCATION_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["LOCATION_HUFFMAN"]
            msg_binary = encode_msg_bin(message, encoding)
        elif ms.issubset(ASCII_Frequencies.G5_addresses.keys()):
            encoding = ADDRESS_HUFFMAN
            scheme_id = self.encoding_to_scheme_id["ADDRESS_HUFFMAN"]
            msg_binary = encode_msg_bin(message, encoding)
        # elif ms.issubset(ASCII_Frequencies.letter_freq_with_space.keys()):
        #     encoding = LETTERS_HUFFMAN
        #     scheme_id = self.encoding_to_scheme_id["LETTERS_HUFFMAN"]
        # elif ms.issubset(ASCII_Frequencies.num_freq.keys()):
        #     encoding = NUMBER_HUFFMAN
        #     scheme_id = self.encoding_to_scheme_id["NUMBER_HUFFMAN"]
        # else: # unlock for the time being
        #     tokens = set(message.split())
            
        #print(message)
        # Calculate checksum before prepending the leading 1 bit
        # assert(len(self.compute_crc16_checksum(msg_huffman_binary)) == 16)
        checksum = self.compute_pearson8_checksum(msg_binary)
        msg_binary = self.compose_huffman_bin(msg_binary, checksum, scheme_id, truncated)
        # print("Encoding Group {}".format(int(scheme_id, 2) + 1))
        cards = bin_to_cards(msg_binary)
        return cards



    def decode(self, deck):
        """
        Given a binary str, use 'decode_bin_msg' to decode it
        see main below
        """
        # print("after shuffling ", deck)
        tmp = []
        for card in deck:
            if card < self.dict_encoding.num_cards_used or card >= 50:
                tmp.append(card)
        if tmp[-2:] == [50, 51]:
            deck = tmp[:-2]
            num = self.dict_encoding.perm_number(deck)
            num = format(num, 'b')
            num, scheme_id, truncated = num[:-4], num[-4:-1], num[-1]
            group_id = int(scheme_id, 2) + 1
            if num != "" and scheme_id != "" and truncated != "" and group_id in self.dict_encoding.group_vocab:
                return self.dict_encoding.decode(deck)
        
        for perm_bound in range(1, 52):
            msg_cards = []
            for c in deck:
                if c <= perm_bound:
                    msg_cards.append(c)
            bin_raw = cards_to_bin(msg_cards)
            if self.decompose_huffman_bin(bin_raw) is not None:
                bin_msg, scheme_id, truncated, checksum = self.decompose_huffman_bin(bin_raw)
                if checksum == self.compute_pearson8_checksum(bin_msg):
                    if scheme_id in self.scheme_id_to_encoding:
                        # print("scheme_id ", scheme_id)
                        decoded_message = decode_bin_msg(bin_msg, self.scheme_id_to_encoding[scheme_id])
                        if truncated == "1":
                            decoded_message = "PARTIAL: " + decoded_message
                        return decoded_message
        return "NULL"


if __name__=='__main__':
    agent = Agent()
    encoded = agent.encode('abcd')
    # print("ENCODED: ", encoded)
    decoded = agent.decode(encoded)
    print('Encoded msg: ', encoded)
    print('Decoded msg: ', decoded)
    