from email.headerregistry import Address
from enum import Enum
import random
import string
import math
from collections import defaultdict
import os
import enchant
import hashlib
import numpy as np

UNTOK = '*'
EMPTY = ''
DICT_SIZE = 27000
SENTENCE_LEN = 6
ENGLISH_DICTIONARY = enchant.Dict("en_US") # pip install pyenchant

# g1 + g4 + g3
# g7 + g6 (1gram)
# g2
# g5
# g8

class Domain_Info():
    def __init__(self):
        '''Store all inforamtion about all domains'''
        self.allDomains = list()
        self.allLayouts = list()
        self.layoutDict = {}
        self.domainDict = {}
        # self.domains = {group1: 0, group2: 1,  group6: 3, group7: 3  }
        # self.dictionary67 = set() # share dictionary between g6 and g7
        # self.dictionary14 = set() # share dictionary between g1 and g4
        self.add_g1_domain()
        self.add_g2_domain()
        self.add_g3_domain()
        self.add_g4_domain()
        self.add_g5_domain()
        self.add_g6_domain()
        self.add_g7_domain()
        self.add_g8_domain()

    def add_g1_domain(self):
        '''Add a new domain to the domain info object'''
        # So far, g1 didnt provide any domain just yet, assume ascii
        domain = " 0123456789"
        domain += "abcdefghijklmnopqrstuvwxyz"
        domain += "."
        domain_list = []
        for c in domain:
            domain_list.append(c)
        self.allDomains.append(domain_list)
        self.allLayouts.append([0])
        # self.domains[Domain.G1] = [domain_list]
        self.layoutDict[1] = [0]

    def add_g2_domain(self):
        '''Add a new domain to the domain info object'''
        # AIRPORT: abbreviation, location, number
        # Char: upper case, digit, space
        # self.domains[Domain.AIRPORT] = dict()
        domain2_list = [[], [], []]
        month = [x for x in range(1, 13)]
        day = [x for x in range(1, 29)]
        year = [x for x in range(2023, 2026)]
        for y in year:
            for m in month:
                if m < 10:
                    m = '0' + str(m)
                for d in day:
                    if d < 10:
                        d = '0' + str(d)
                    time = str(m) + str(d) + str(y)
                    domain2_list[-1].append(time)
        domain2_list[1].append('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # self.domains[Domain.AIRPORT].append(time)
        # for i1 in (string.ascii_uppercase + string.digits):
        #     for i2 in (string.ascii_uppercase + string.digits):
        #         for i3 in (string.ascii_uppercase + string.digits):
        #             for i4 in (string.ascii_uppercase + string.digits):
        #                 self.domains[Domain.AIRPORT].append(i1+i2+i3+i4)
        with open("./messages/agent2/airportcodes.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain2_list[0].append(line)
                # self.domains[Domain.AIRPORT].append(line)
                line = f.readline()
        self.allDomains.append(domain2_list)
        self.allLayouts.append([0, 1, 1, 1, 1, 2])
        self.layoutDict[2] = [0, 1, 1, 1, 1, 2]

    def add_g3_domain(self):
        '''Add a new domain to the domain info object'''
        # PASSWORD: @, number/words combination
        # Char: upper/lower case, number, "@"
        domain3_list = []
        domain = "0123456789"
        domain += "abcdefghijklmnopqrstuvwxyz"
        for c in domain:
            domain3_list.append(c)
        self.allDomains.append(domain3_list)
        self.allLayouts.append([0])
        self.layoutDict[3] = [0]

    def add_g4_domain(self):
        '''Add a new domain to the domain info object'''
        # Coordinate: latitude, longitude
        # Char: digit, N, E, S, W, ".", ",", space
        domain4_list = [[], [], []]  # Num before ., Num after ., NEWS
        domain4_list[2].append('NEWS')
        for i in range(0, 180):
            domain4_list[0].append(str(i))
        for i in range(0, 10000):
            domain4_list[1].append(str('{:04}'.format(i)))
        self.allDomains.append(domain4_list)
        self.allLayouts.append([0, 1, 2, 0, 1, 2])
        self.layoutDict[4] = [0, 1, 2, 0, 1, 2]

    def add_g5_domain(self):
        '''Add a new domain to the domain info object'''
        # Address: Number, Street suffix, streetname
        # Char: words, digit, space, ",", ".", "/"
        # self.domains[Domain.ADDRESS] = []
        domain5_list = [[], [], []]
        # self.domains[Domain.ADDRESS] = [[],[],[]]  #Number, Streetname, Street Suffix
        domain5_layout = [0, 1, 2]  # Number, Streetname, Street Suffix
        with open("./messages/agent5/street_name.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain5_list[1].append(line)
                line = f.readline()
        with open("./messages/agent5/street_suffix.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain5_list[2].append(line)
                line = f.readline()
        for i in range(10000):
            i = str(i)
            domain5_list[0].append(i)
        self.allDomains.append(domain5_list)
        self.allLayouts.append([0, 1, 2])
        self.layoutDict[5] = [0, 1, 2]

    def add_g6_domain(self):
        '''Add a new domain to the domain info object'''
        # N-gram: 1-gram, 2-gram, ..., 9-gram
        # Char: words, digit, space

        domain67_list = set()  # IMPORTANT TO AVOID DUPLICATES
        with open("./messages/agent6/corpus-ngram-" + str(1) + ".txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain67_list.add(line)
                line = f.readline()
        with open("./messages/agent7/30k.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()

                # if ENGLISH_DICTIONARY.check(line): #TODO fix pyenchant
                domain67_list.add(line)
                line = f.readline()

        self.allDomains.append(domain67_list)  # for group 6
        self.allLayouts.append([0])
        self.layoutDict[6] = [0]

    def add_g7_domain(self):
        '''Add a new domain to the domain info object'''
        # Duh, this is the domain we are working on
        domain67_list = set()  # IMPORTANT TO AVOID DUPLICATES
        with open("./messages/agent6/corpus-ngram-" + str(1) + ".txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain67_list.add(line)
                line = f.readline()
        with open("./messages/agent7/30k.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()

                # if ENGLISH_DICTIONARY.check(line):
                domain67_list.add(line)
                line = f.readline()

        self.allDomains.append(domain67_list)  # for group 7
        self.allLayouts.append([0])
        self.layoutDict[7] = [0]

    def add_g8_domain(self):
        '''Add a new domain to the domain info object'''
        # Name/Places: Name, places
        # Char: upper/lower case, space
        domain8_list = []
        with open("./messages/agent8/names.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain8_list.append(line)
                line = f.readline()

        with open("./messages/agent8/places.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain8_list.append(line)
                line = f.readline()
        self.allDomains.append(domain8_list)
        self.allLayouts.append([0])  # TODO
        self.layoutDict[8] = [0]

    def get_domain(self, domain_index):
        '''Get domain object from domain index'''
        return self.domains[domain_index]


class Domain(Enum):
    G1 = 1
    AIRPORT = 2
    PASSWORD = 3
    LOCATION = 4
    ADDRESS = 5
    NGRAM = 6
    DICTIONARY = 7
    NAME_PLACES = 8


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


'''def composed_of_words(msg, dictionary):
    """Recursively check if str is composed of words from dictionary."""
    for i in range(len(msg) + 1):
        if msg[:i].strip() in dictionary:
            if i == len(msg):
                return True
            if not composed_of_words(msg[i:].strip(), dictionary):
                continue
            else:
                return True
    return False'''


class Domain_Classifier():
    def __init__(self) -> None:
        self.domain_info = Domain_Info()

    def is_password(self, msg):
        if msg[0] != "@":
            return False
        return msg[1:]

    def is_location(self, msg):
        if len(msg.split(",")) != 2:
            return False
        msg = msg.strip()
        tokens = []
        c1, c2 = msg.split(",")
        if (c1[-1] == "N" or c1[-1] == "S") and (c2[-1] == "E" or c2[-1] == "W"):
            for v in [c1, c2]:
                left, right = v.split('.')
                lspace, rspace = right.split(' ')
                tokens.append(left)
                tokens.append(lspace)
                tokens.append(rspace)
            return tokens
        return False

    def is_airport(self, msg):
        if len(msg.split(" ")) != 3:
            return False
        orig_msg = msg
        msg = msg.split(" ")
        airport_code = msg[0]

        if (airport_code in self.domain_info.allDomains[1][0]
                and msg[2].isdigit()):
            return orig_msg.split(' ')
        return False

    def is_address(self, msg):
        msg = msg.strip()
        partition = msg.split(" ")
        if len(partition) < 2:
            return False
        if not partition[0].isdigit():
            return False
        if partition[1] not in self.domain_info.allDomains[4][1]:
            return False
        else:
            return msg.split(" ")

    def is_name_places(self, msg):
        if not msg[0].isupper():
            return False
        for x in msg.strip().split(" "):
            if x not in self.domain_info.allDomains[7]:
                return False
        return msg.split(" ")

    def is_dictionary(self, msg):
        if len(msg.split()) > 6 or has_numbers(msg):
            return False

        msg = msg.strip()
        for m in msg.split():
            if m not in self.domain_info.allDomains[6]:
                return False

        return msg.split(' ')

    def is_ngram(self, msg):

        msg = msg.strip()
        for m in msg.split():
            if m not in self.domain_info.allDomains[6]:
                return False

        return msg.split(' ')

    def predict(self, msg):
        """
        Classifies the message into one of the following domains:
        - G1 (generic/default) *
        - AIRPORT *
        - PASSWORD *
        - LOCATION *
        - ADDRESS *
        - NGRAM
        - DICIONARY *
        - NAME_PLACES *
        """
        if self.is_password(msg):
            return Domain.PASSWORD, self.is_password(msg)
        elif self.is_location(msg):
            return Domain.LOCATION
        elif self.is_airport(msg):
            return Domain.AIRPORT
        elif self.is_address(msg):
            return Domain.ADDRESS
        elif self.is_name_places(msg):
            return Domain.NAME_PLACES
        elif self.is_dictionary(msg):
            return Domain.DICTIONARY
        elif self.is_ngram(msg):
            return Domain.NGRAM
        return Domain.G1 # default ascii


    # def binary_predict(self, msg):
    #     if self.is_dictionary(msg):
    #         return Domain.DICTIONARY
    #     return Domain.G1

ENCODING_MAX_LENGTH = 30
MAX_TOKENS = 14
NUM_DOMAINS = 6
META_LENGTH = 7


# TODO: split into Encoder and Decoder - the relevant methods for each class are marked
class EncoderDecoder:
    # def __init__(self, domain_idx = 0, domains = ()):
    # TODO: If the logic is moved from Agent to Encoder and Decoder, these parameters should be retrievable from the message or deck
    def __init__(self, classifier = Domain_Classifier(), all_domains = Domain_Info()):
        self.classifier = classifier
        self.all_domains = all_domains

        # self.domain_idx = domain_idx
        # self.indices = domains
        # self.dict_sizes = [len(domain) for domain in domains]
        # self.dictionaries = []
        self.domain_idx = None
        self.indices = None
        self.dict_sizes = None
        self.dictionaries = None

        factorials = [0] * 52
        for i in range(52):
            factorials[i] = math.factorial(52 - i - 1)
        self.factorials_reverse = factorials

    # TODO: Decoder
    def perm_number(self, permutation):
        n = len(permutation)
        factorials = self.factorials_reverse[-n:]
        number = 0
        for i in range(n):
            k = 0
            for j in range(i + 1, n):
                if permutation[j] < permutation[i]:
                    k += 1
            number += k * factorials[i]
        return number

    # TODO: Decoder
    def nth_perm(self, n, perm_zero):
        perm = []
        items = perm_zero[:]
        factorials = self.factorials_reverse[-len(items):]

        for f in factorials:
            lehmer = n // f
            x = items.pop(lehmer)
            perm.append(x)
            n %= f
        return perm

    # TODO: Encoder
    # TODO: change to work with factorial sums - could be noticeable for small factorials
    def get_encoding_length(self, tokens, layout):
        # if len(self.dict_sizes) == 1:
        #     token_permutation_size = self.dict_sizes[0] ** len(tokens)
        # else:
        token_permutation_size = int(np.prod([self.dict_sizes[idx] for idx in layout]))

        for i in range(2, ENCODING_MAX_LENGTH):
            if math.factorial(i) > token_permutation_size:
                return i

        # message too long
        return 0

    # TODO: Encoder
    # methods to index or retrieve the index of varying domains
    @staticmethod
    def tree_index(factors, max_factors):
        index = 0
        for i in range(len(factors)):
            index += factors[i] * int(np.prod(max_factors[i+1:]))

        return index

    # TODO: Decoder
    @staticmethod
    def tree_factors(index, max_factors):
        factors = []
        for i in range(len(max_factors)):
            n_children = int(np.prod(max_factors[i+1:]))
            factor = index//n_children
            factors.append(factor)
            index %= n_children

        return factors

    # TODO: Encoder
    # get metadata permutation (6 cards currently)
    def encode_metadata(self, encoding_len, num_tokens, domain_idx, partial):
        factors = [encoding_len, num_tokens, domain_idx, partial]
        max_factors = [ENCODING_MAX_LENGTH, MAX_TOKENS, NUM_DOMAINS, 2] # partial is just a flag that can only be 0 or 1
        meta_idx = self.tree_index(factors, max_factors)
        meta_perm_zero = list(range(52-META_LENGTH, 52))
        meta_perm = self.nth_perm(meta_idx, meta_perm_zero)

        return meta_perm

    # TODO: Decoder
    def decode_metadata(self, meta_perm):
        max_factors = [ENCODING_MAX_LENGTH, MAX_TOKENS, NUM_DOMAINS, 2]
        meta_idx = self.perm_number(meta_perm)
        factors = self.tree_factors(meta_idx, max_factors)

        return factors

    # TODO: Encoder
    # encode a message which is a list of tokens
    def encode(self, tokens, layout):
        encoding_len = self.get_encoding_length(tokens, layout)
        partial = 0
        while encoding_len == 0:
            partial = 1
            tokens.pop()
            layout.pop()
            encoding_len = self.get_encoding_length(tokens, layout)

        max_indices = [self.dict_sizes[dict_idx] for dict_idx in layout]
        word_indices = [self.dictionaries[dict_idx][token] for token, dict_idx in zip(tokens, layout)]

        # get the index of the message
        num = self.tree_index(word_indices, max_indices)
        perm_zero = list(range(52-META_LENGTH-ENCODING_MAX_LENGTH, 52-META_LENGTH))
        perm = self.nth_perm(num, perm_zero)[::-1]
        metadata_perm = self.encode_metadata(encoding_len, self.domain_idx, len(tokens), partial)

        deck = list(range(0, 52-META_LENGTH-encoding_len)) + perm + metadata_perm
        print(deck)
        return deck

    # TODO: Decoder
    # returns a list of tokens TODO: WIP: add calculations based on factorial sums from get_encoding_length()
    def decode(self, deck):
        metadata_perm = []
        for card in deck:
            if 52-META_LENGTH < card < 52:
                metadata_perm.append(card)
        message_len, domain, partial = self.decode_metadata(metadata_perm)

        message_perm = []
        for card in deck:
            if 52-message_len-META_LENGTH < card < 52-META_LENGTH:
                message_perm.append(card)

        message_perm = message_perm[::-1]
        message_perm_num = self.perm_number(message_perm)

        layout = self.
        max_factors =
        token_indices = self.get
        # TODO: WIP

    # TODO: BOTH Encoder and Decoder (maybe move it outside of the class? static anyway)
    @staticmethod
    def get_layout(num_tokens, did):
        if did == 1:
            return [0] * num_tokens
        elif did == 2:
            return [0, 1, 1, 1, 1, 2]
        elif did == 3:
            return [0] * num_tokens
        elif did == 4:
            return [0, 1, 2, 0, 1, 2]
        elif did == 5:
            return [0, 1, 2]
        elif did == 6:
            return [0] * num_tokens
        elif did == 7:
            return [0] * num_tokens
        else:
            return [0] * num_tokens



class Agent:
    def __init__(self, encoding_len=26):
        self.classifier = Domain_Classifier()
        self.all_domains = Domain_Info()

    # TODO: move all the logic into Encoder (maybe make a "process message" method that classifies, tokenizes, and retrieves domains)
    def encode(self, message):
        ed = EncoderDecoder(self.classifier, self.all_domains)
        domain_id = self.classifier.predict(message)
        if domain_id == 3:
            return
        layout = self.all_domains.layoutDict[domain_id]
        domains = self.all_domains.domainDict[domain_id]
        tokens = self.classifier.tokenize(message, domain_id)
        # ed = EncoderDecoder(domain_id, domains)

        return ed.encode(tokens, layout)

    # TODO: move all the logic into Decoder (maybe add a "process tokens" method that assembles the tokens into original message based on domain)
    def decoder(self, deck):
        ed = EncoderDecoder(self.classifier, self.all_domains)
        return ed.decode(deck)






    # def encode(self, message):
    #     print('Encoding "', message, '"')
    #     domain = self.classifier
    #     message = ' '.join(message.split()[:6])
    #     x  = self.ed.str_to_num(message)
    #     checksum = self.ed.set_checksum(x)
    #     checksum_cards = self.perm_ck.nth_perm(checksum)
    #     a = list(range(46 - self.encoding_len))
    #     b = self.ed.str_to_perm(message)
    #     c =checksum_cards
    #     encoded_deck = a+b+c
    #     print('Encoded deck:\n', encoded_deck, '\n---------')
    #     return encoded_deck

    def decode(self, deck):
        msg_perm = []
        checksum = []
        for card in deck:
            if 20 <= card <= 45:
                msg_perm.append(card)
            if card > 45:
                checksum.append(card)

        #print('\nMessage Cards:', msg_perm)
        #print('Checksum Cards:', checksum)
        msg_num = self.ed.perm_number(msg_perm)

        decoded_checksum = self.perm_ck.perm_number(checksum)
        message_checksum = self.perm_ck.set_checksum(msg_num)        



def test_classifier():
    # - ASCII
    # - AIRPORT
    # - PASSWORD
    # - LOCATION
    # - ADDRESS
    # - NGRAM
    # - DICIONARY
    # - NAME_PLACES
    classifier = Domain_Classifier()
    # g7_dict = Domain_Classifier().domain_info.get_domain(Domain.DICTIONARY)
    # g6_dict = Domain_Classifier().domain_info.get_domain(Domain.NGRAM)[1]
    # overall = g6_dict.union(set(g7_dict))
    # print(len(g7_dict), len(g6_dict), len(overall))
    # msg = "@pneumatoscope9mesorrhinium5worklessness489"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "1.1714 S, 36.8356 E"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "SWF LX3M 12032025"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "the of a apple orange kills"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "5 East Main Meadow "
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "escalation encourage least kyiv ukrainian environment defeat"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "As of Monday October 31 Putin had not signed the decree required to officially end mobilization"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "2uez4cw6tc4"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    # msg = "Adwoa Abdullatif Zeona Zephyr"
    # print(f"{msg:>100} -> {classifier.predict(msg)}")
    
    for i in range(1, 9):
        with open(f"./test_classifier/g{i}_example.txt") as f:
            msg = f.readline().strip()
            while msg:
                prediction = classifier.binary_predict(msg)
                print(prediction, i)
                msg = f.readline().strip()
    return

test_classifier()


# class EncoderDecoderDict:
    # def __init__(self, n):
    #     self.encoding_len = n
    #     if n < 7: #If less than 7 bits its for checksum
    #         self.perm_zero = [46,47,48,49,50,51]
    #     else:
    #         self.perm_zero = list(range(46-n, 46)) #[20,21,...45]
    #     self.max_messge_length = 12 #TODO TEST AND CHANGE THIS VALUE
    #     factorials = [0] * n
    #     for i in range(n):
    #         factorials[i] = math.factorial(n-i-1)
    #     self.factorials = factorials
    #
    #     words_dict = {EMPTY: 0}
    #     words_index = [EMPTY]
    #     with open(os.path.dirname(__file__) + '/../messages/agent7/30k.txt') as f:
    #         for i in range(1, DICT_SIZE-1):     # reserve 1 for empty and 1 for unknown
    #             word = f.readline().rstrip('\t\n')
    #             words_dict[word] = i
    #             words_index.append(word)
    #     words_dict[UNTOK] = DICT_SIZE-1
    #     words_index.append(UNTOK)
    #     self.words_dict = words_dict
    #     self.words_index = words_index
    #
    # def perm_number(self, permutation):
    #     n = len(permutation)
    #     # s = sorted(permutation)
    #     number = 0
    #     for i in range(n):
    #         k = 0
    #         for j in range(i + 1, n):
    #             if permutation[j] < permutation[i]:
    #                 k += 1
    #         number += k * self.factorials[i]
    #     return number
    #
    # def nth_perm(self, n):
    #     perm = []
    #     items = self.perm_zero[:]
    #
    #     for f in self.factorials:
    #         lehmer = n // f
    #         x = items.pop(lehmer)
    #         perm.append(x)
    #         n %= f
    #     return perm
    #
    # def str_to_perm(self, message):
    #     tokens = message.split()
    #     init = [EMPTY for i in range(SENTENCE_LEN)]
    #     for i in range(len(tokens)):
    #         init[i] = tokens[i]
    #     tokens = init[::-1]
    #     num = 0
    #     for i in range(SENTENCE_LEN):
    #         num += self.words_dict.get(tokens[i], DICT_SIZE-1) * DICT_SIZE**i
    #     return self.nth_perm(num)
    #
    # def str_to_num(self, message):
    #     tokens = message.split()
    #     init = [EMPTY for i in range(SENTENCE_LEN)]
    #     for i in range(len(tokens)):
    #         init[i] = tokens[i]
    #     tokens = init[::-1]
    #     num = 0
    #     for i in range(SENTENCE_LEN):
    #         num += self.words_dict.get(tokens[i], DICT_SIZE-1) * DICT_SIZE**i
    #     return num
    #
    # def perm_to_str(self, perm):
    #     num = self.perm_number(perm)
    #     words = []
    #     for i in range(SENTENCE_LEN):
    #         index = num % DICT_SIZE
    #         words.append(self.words_index[index])
    #         num = num // DICT_SIZE
    #     return ' '.join(words[::-1]).strip()
    #
    # def set_checksum(self, num, base=10):
    #     num_bin = bin(num)[2:]
    #     chunk_len = 5
    #     checksum = 0
    #     mod_prime = 113
    #     while len(num_bin) > 0:
    #         bin_chunk = num_bin[:chunk_len]
    #         num_bin = num_bin[chunk_len:]
    #
    #         num_chunk = int(bin_chunk, 2)
    #         checksum = ((checksum + num_chunk) * base) % mod_prime
    #     return checksum