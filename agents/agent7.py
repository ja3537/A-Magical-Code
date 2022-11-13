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
        self.all_lists = list()
        self.all_dicts = list()
        self.group_to_lists = []

        self.add_g1_domain()
        self.add_g2_domain()
        self.add_g3_domain()
        self.add_g4_domain()
        self.add_g5_domain()
        self.add_g6_domain()
        self.add_g7_domain()
        self.add_g8_domain()

        for domain in self.all_lists:
            dict_dom = []
            for li in domain:
                word_to_idx = {}
                for i in range(len(li)):
                    word_to_idx[li[i]] = i
                dict_dom.append(word_to_idx)
            self.all_dicts.append(dict_dom)


    def add_g1_domain(self):
        '''Add a new domain to the domain info object'''
        # So far, g1 didnt provide any domain just yet, assume ascii
        domain = " 0123456789"
        domain += "abcdefghijklmnopqrstuvwxyz"
        domain += "."
        domain_list = []
        for c in domain:
            domain_list.append(c)
        self.all_lists.append([domain_list])
        self.group_to_lists.append(0)

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
        self.all_lists.append(domain2_list)
        self.group_to_lists.append(1)

    def add_g3_domain(self):
        '''Add a new domain to the domain info object'''
        # PASSWORD: @, number/words combination
        # Char: upper/lower case, number
        # domain3_list = []
        # domain = "0123456789"
        # domain += "abcdefghijklmnopqrstuvwxyz"
        # for c in domain:
        #     domain3_list.append(c)
        # self.allDomains.append([domain3_list])
        self.group_to_lists.append(0)

    def add_g4_domain(self):
        '''Add a new domain to the domain info object'''
        # Coordinate: latitude, longitude
        # Char: digit, N, E, S, W
        domain4_list = [[], [], []]  # Num before ., Num after ., NEWS
        domain4_list[2].append('NEWS')
        for i in range(0, 180):
            domain4_list[0].append(str(i))
        for i in range(0, 10000):
            domain4_list[1].append(str('{:04}'.format(i)))
        self.all_lists.append(domain4_list)
        self.group_to_lists.append(2)


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
        self.all_lists.append(domain5_list)
        self.group_to_lists.append(3)

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
        with open("./messages/agent7/words.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                if ENGLISH_DICTIONARY.check(line):
                    domain67_list.add(line)
                line = f.readline()

        self.all_lists.append([list(domain67_list)])  # for group 6
        self.group_to_lists.append(4)

    def add_g7_domain(self):
        '''Add a new domain to the domain info object'''
        # Duh, this is the domain we are working on
        self.group_to_lists.append(4)

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
        self.all_lists.append([domain8_list])
        self.group_to_lists.append(5)


class Domain(Enum):
    G1 = 0
    AIRPORT = 1
    PASSWORD = 2
    LOCATION = 3
    ADDRESS = 4
    NGRAM = 5
    DICTIONARY = 6
    NAME_PLACES = 7


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

        if (airport_code in self.domain_info.all_lists[1][0]
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
        if partition[1] not in self.domain_info.all_lists[3][1]:
            return False
        else:
            return msg.split(" ")

    def is_name_places(self, msg):
        if not msg[0].isupper():
            return False
        for x in msg.strip().split(" "):
            if x not in self.domain_info.all_lists[5][0]:
                return False
        return msg.split(" ")

    def is_dictionary(self, msg):
        print(msg)
        print(len(self.domain_info.all_lists))
        if len(msg.split()) > 6 or has_numbers(msg):
            return False

        msg = msg.strip()
        for m in msg.split():
            if m not in self.domain_info.all_lists[4][0]:
                return False

        return msg.split(' ')

    def is_ngram(self, msg):

        msg = msg.strip()
        for m in msg.split():
            if m not in self.domain_info.all_lists[4][0]:
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
        if did == 0:
            return [0] * num_tokens
        elif did == 1:
            return [0, 1, 1, 1, 1, 2]
        elif did == 2:
            return [0] * num_tokens
        elif did == 3:
            return [0, 1, 2, 0, 1, 2]
        elif did == 4:
            return [0, 1, 2]
        elif did == 5:
            return [0] * num_tokens
        elif did == 6:
            return [0] * num_tokens
        else:
            return [0] * num_tokens

class Encoder:
    def __init__(self, classifier = Domain_Classifier(), all_domains = Domain_Info()) -> None:
        self.classifier = classifier
        self.all_domains = all_domains

        factorials = [0] * 52
        for i in range(52):
            factorials[i] = math.factorial(52 - i - 1)
        self.factorials_reverse = factorials
    
    def encode(self, msg):
        domain_id, tokens = self.classifier.predict(msg)
        layout = get_layout(len(tokens), domain_id)
        word_to_index = self.all_domains.all_dicts[self.all_domains.group_to_lists[domain_id.value]]
        dict_sizes = [len(d) for d in word_to_index]
        word_indices = [word_to_index[dict_idx][token] for token, dict_idx in zip(tokens, layout)]
        max_indices = [dict_sizes[dict_idx] for dict_idx in layout]
        encoding_len, perm_idx = self.get_encoding_length(word_indices, max_indices)
        partial = 0

        while encoding_len == -1:
            partial = 1
            word_indices.pop()
            layout.pop()
            max_indices.pop()
            encoding_len, perm_idx = self.get_encoding_length(word_indices, max_indices)

        # get the index of the message
        # num = self.tree_index(word_indices, max_indices)
        perm_zero = list(range(52-META_LENGTH-ENCODING_MAX_LENGTH, 52-META_LENGTH))
        perm = self.nth_perm(perm_idx, perm_zero)[::-1]
        metadata_perm = self.encode_metadata(encoding_len, len(tokens), domain_id.value, partial)
        deck = list(range(0, 52-META_LENGTH-encoding_len)) + perm + metadata_perm

        return deck

    def get_encoding_length(self, indices, max_indices):
        num = self.tree_index(indices, max_indices)
        if num == 0:
            return 0, 0

        s = 1
        for i in range(1, ENCODING_MAX_LENGTH+1):
            t = s
            s += math.factorial(i)
            if s >= num:
                card_deck_idx = num - t
                return i, card_deck_idx

        # message too long
        return -1, -1

    # methods to index or retrieve the index of varying domains
    @staticmethod
    def tree_index(factors, max_factors):
        index = 0
        for i in range(len(factors)):
            index += factors[i] * int(np.prod(max_factors[i+1:]))

        return index

    # get metadata permutation (6 cards currently)
    def encode_metadata(self, encoding_len, num_tokens, domain_idx, partial):
        # TODO: factors doesnt added up to decode metadata
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

def assemble_airport(tokens):
    if len(tokens) == 1:
        return '{}'.format(*tokens)
    elif len(tokens) == 2:
        return '{} {}'.format(*tokens)
    elif len(tokens) == 3:
        return '{} {}{}'.format(*tokens)
    elif len(tokens) == 4:
        return '{} {}{}{}'.format(*tokens)
    elif len(tokens) == 5:
        return '{} {}{}{}{}'.format(*tokens)
    return '{} {}{}{}{} {}'.format(*tokens)

def assemble_location(tokens):
    if len(tokens) == 1:
        return '{}'.format(*tokens)
    elif len(tokens) == 2:
        return '{}.{}'.format(*tokens)
    elif len(tokens) == 3:
        return '{}.{} {},'.format(*tokens)
    elif len(tokens) == 4:
        return '{}.{} {}, {}.'.format(*tokens)
    elif len(tokens) == 5:
        return '{}.{} {}, {}.{}'.format(*tokens)
    return '{}.{} {}, {}.{} {}'.format(*tokens)

def assemble_message(tokens, domain_id):
    if domain_id == 0:
        return ''.join(tokens)
    elif domain_id == 1:
        return assemble_airport(tokens)
    elif domain_id == 2:
        return '@' + ''.join(tokens)
    elif domain_id == 3:
        return assemble_location(tokens)
    else:   # the rest of the groups
        return ' '.join(tokens)


class Decoder:
    def __init__(self, all_domains = Domain_Info()) -> None:
        self.all_domains = all_domains
        
        factorials = [0] * 52
        for i in range(52):
            factorials[i] = math.factorial(52 - i - 1)
        self.factorials_reverse = factorials
    
    def decode(self, deck):
        metadata_perm = []
        for card in deck:
            if 52-META_LENGTH < card < 52:
                metadata_perm.append(card)
        factors = self.decode_metadata(metadata_perm)
        #factors = [1, 6, 6, 0] # Hard code for now
        # TODO: factors doesnt added up to encode metadata
        encoding_len, message_len, domain_id, partial = factors

        message_perm = []
        for card in deck:
            if 52-encoding_len-META_LENGTH < card < 52-META_LENGTH:
                message_perm.append(card)

        message_perm = message_perm[::-1]

        if not message_perm:
            message_perm_num = 0
        else:
            message_perm_num = self.perm_number(message_perm)

        # reconstruct original index from the factorial sum
        actual_num = sum([math.factorial(i) for i in range(encoding_len)]) + message_perm_num
        layout = get_layout(message_len, domain_id)
        index_to_word = self.all_domains.all_dicts[self.all_domains.group_to_lists[domain_id]]
        dict_sizes = [len(d) for d in index_to_word]
        max_factors = [dict_sizes[dict_idx] for dict_idx in layout]
        word_indices = self.tree_factors(actual_num, max_factors)
        # TODO:      
        tokens = []
        for word_index, dict_idx in zip(word_indices, layout):
            try:
                tokens.append(index_to_word[dict_idx][word_index])
            except:
                partial=True
                break
        #tokens = [index_to_word[dict_idx][word_index] for word_index, dict_idx in zip(word_indices, layout)]
        original_message = assemble_message(tokens, domain_id)

        return 'PARTIAL: ' if partial else '' + original_message

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
        
    def encode(self, message):
        return self.encoder.encode(message)

    def decode(self, deck):
        return self.decoder.decode(deck)


def test_encoder_decoder():
    agent = Agent()
    msg = 'vegetative macho sob elaborated reeve embellishments'
    deck = agent.encode(msg)
    msg = agent.decode(deck)
    print(msg)

test_encoder_decoder()