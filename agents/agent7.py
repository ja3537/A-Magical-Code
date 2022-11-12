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

class Domain_Info():
    def __init__(self):
        '''Store all inforamtion about all domains'''
        self.allDomains = list()
        self.allLayouts = list()
        self.layoutDict = {}
        #self.domainDict = {}
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
        self.layoutDict[Domain.G1] = [0]

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
        with open("./messages/agent2/airportcodes.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                domain2_list[0].append(line)
                # self.domains[Domain.AIRPORT].append(line)
                line = f.readline()
        self.allDomains.append(domain2_list)
        self.allLayouts.append([0, 1, 1, 1, 1, 2])
        self.layoutDict[Domain.AIRPORT] = [0, 1, 1, 1, 1, 2]

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
        self.layoutDict[Domain.PASSWORD] = [0]

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
        self.layoutDict[Domain.LOCATION] = [0, 1, 2, 0, 1, 2]

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
        self.layoutDict[Domain.ADDRESS] = [0, 1, 2]

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
        self.layoutDict[Domain.NGRAM] = [0]

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
        self.layoutDict[Domain.DICTIONARY] = [0]

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
        self.layoutDict[Domain.NAME_PLACES] = [0]


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

    def is_g1(self, msg):
        return [x for x in msg]
    
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
            return Domain.LOCATION, self.is_location(msg)
        elif self.is_airport(msg):
            return Domain.AIRPORT, self.is_airport(msg)
        elif self.is_address(msg):
            return Domain.ADDRESS, self.is_address(msg)
        elif self.is_name_places(msg):
            return Domain.NAME_PLACES, self.is_name_places(msg)
        elif self.is_dictionary(msg):
            return Domain.DICTIONARY, self.is_dictionary(msg)
        elif self.is_ngram(msg):
            return Domain.NGRAM, self.is_ngram(msg)
        return Domain.G1, self.is_g1(msg) # default ascii

ENCODING_MAX_LENGTH = 30
MAX_TOKENS = 14
NUM_DOMAINS = 6
META_LENGTH = 7

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

class Encoder():
    def __init__(self, classifier = Domain_Classifier(), all_domains = Domain_Info()) -> None:
        self.classifier = classifier
        self.all_domains = all_domains
        # TODO Initialize these
        self.domain_idx = None
        self.indices = None
        self.dict_sizes = None
        self.dictionaries = None

        factorials = [0] * 52
        for i in range(52):
            factorials[i] = math.factorial(52 - i - 1)
        self.factorials_reverse = factorials
    
    def encode(self, msg):
        domain_id, tokens = self.classifier.predict(msg)
        if domain_id == 3:
            return
        layout = self.all_domains.layoutDict[domain_id]
        #domains = self.all_domains.domainDict[domain_id]
        
        encoding_len = self.get_encoding_length(tokens, layout)
        partial = 0
        # ????
        while encoding_len == 0:
            partial = 1
            tokens.pop()
            layout.pop()
            encoding_len = self.get_encoding_length(tokens, layout)

        # self.dict_sizes not initialized
        max_indices = [self.dict_sizes[dict_idx] for dict_idx in layout]
        word_indices = [self.dictionaries[dict_idx][token] for token, dict_idx in zip(tokens, layout)]

        # get the index of the message
        num = self.tree_index(word_indices, max_indices)
        perm_zero = list(range(52-META_LENGTH-ENCODING_MAX_LENGTH, 52-META_LENGTH))
        perm = self.nth_perm(num, perm_zero)[::-1]
        metadata_perm = self.encode_metadata(encoding_len, domain_id.value, len(tokens), partial)

        deck = list(range(0, 52-META_LENGTH-encoding_len)) + perm + metadata_perm
        print(deck)
        return deck
    
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

    # methods to index or retrieve the index of varying domains
    @staticmethod
    def tree_index(factors, max_factors):
        index = 0
        for i in range(len(factors)):
            index += factors[i] * int(np.prod(max_factors[i+1:]))

        return index

    # get metadata permutation (6 cards currently)
    def encode_metadata(self, encoding_len, num_tokens, domain_idx, partial):
        factors = [encoding_len, num_tokens, domain_idx, partial]
        max_factors = [ENCODING_MAX_LENGTH, MAX_TOKENS, NUM_DOMAINS, 2] # partial is just a flag that can only be 0 or 1
        meta_idx = self.tree_index(factors, max_factors)
        meta_perm_zero = list(range(52-META_LENGTH, 52))
        meta_perm = self.nth_perm(meta_idx, meta_perm_zero)

        return meta_perm
    
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
    

class Decoder():
    def __init__(self, all_domains = Domain_Info()) -> None:
        self.all_domains = all_domains
        # TODO Initialize these
        self.domain_idx = None
        self.indices = None
        self.dict_sizes = None
        self.dictionaries = None

        factorials = [0] * 52
        for i in range(52):
            factorials[i] = math.factorial(52 - i - 1)
        self.factorials_reverse = factorials
    
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
        
        layout = None
        max_factors = None
        token_indices = None
        # TODO: WIP

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

    def decode_metadata(self, meta_perm):
        max_factors = [ENCODING_MAX_LENGTH, MAX_TOKENS, NUM_DOMAINS, 2]
        meta_idx = self.perm_number(meta_perm)
        factors = self.tree_factors(meta_idx, max_factors)

        return factors
    
    @staticmethod
    def tree_factors(index, max_factors):
        factors = []
        for i in range(len(max_factors)):
            n_children = int(np.prod(max_factors[i+1:]))
            factor = index//n_children
            factors.append(factor)
            index %= n_children

        return factors


# Agent====================================================================

class Agent:
    def __init__(self, encoding_len=26):
        self.classifier = Domain_Classifier()
        self.all_domains = Domain_Info()
        self.encoder = Encoder(self.classifier, self.all_domains)
        self.decoder = Decoder(self.all_domains)
        
    # TODO: move all the logic into Encoder (maybe make a "process message" method that classifies, tokenizes, and retrieves domains)
    def encode(self, message):
        return self.encoder.encode(message)

    # TODO: move all the logic into Decoder (maybe add a "process tokens" method that assembles the tokens into original message based on domain)
    def decode(self, deck):
        return self.decoder.decode(deck)


def test_encoder_decoder():
    agent = Agent()
    msg = 'this concrete house appears extremely gigantic'
    deck = agent.encode(msg)
    msg = agent.decode(deck)
    print(msg)

test_encoder_decoder()
