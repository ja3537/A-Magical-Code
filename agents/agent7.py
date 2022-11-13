# Aditya Kulkarni, Zongyu Chen, Dmitrii Zakharov
from email.headerregistry import Address
from enum import Enum
import random
import string
import math
from collections import defaultdict
import os
import hashlib
import numpy as np

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
        for idx, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
            domain2_list[1].append(c)
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
        for i in 'NEWS':
            domain4_list[2].append(i)
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
            s = set()
            while line:
                line = line.strip()
                s.add(line)
                line = f.readline()
            domain5_list[1] = list(s)
        with open("./messages/agent5/street_suffix.txt", "r") as f:
            line = f.readline()
            s = set()
            while line:
                line = line.strip()
                s.add(line)
                line = f.readline()
            domain5_list[2] = list(s)
        for i in range(0, 10000): 
            domain5_list[0].append(str(i))
            while len(str(i)) < 4:
                i = '0' + str(i)
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
                # if ENGLISH_DICTIONARY.check(line):
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
        s = set()
        with open("./messages/agent8/names.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                s.add(line)
                line = f.readline()
        with open("./messages/agent8/places.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                s.add(line)
                line = f.readline()
        self.all_lists.append([list(s)])
        self.group_to_lists.append(5)


class Domain(Enum):
    G1 = 0
    AIRPORT = 1
    PASSWORD = 2
    LOCATION = 3
    ADDRESS = 4
    NGRAM = 5
    DICTIONARY = 5  # same as NGRAM
    NAME_PLACES = 7


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


class Domain_Classifier():
    def __init__(self) -> None:
        self.domain_info = Domain_Info()

    def is_password(self, msg):
        if msg[0] != "@":
            return False
        return [x for x in msg[1:]]

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
            return [x.strip() for x in tokens]
        return False

    def is_airport(self, msg):
        if len(msg.split(" ")) != 3:
            return False
        orig_msg = msg
        msg = msg.split(" ")
        airport_code = msg[0]

        if (airport_code in self.domain_info.all_lists[1][0]
                and msg[2].isdigit()):
            partition = orig_msg.split(' ')
            return_tokens = []
            return_tokens.append(partition[0])
            for i in partition[1]:
                return_tokens.append(i)
            return_tokens.append(partition[2])
            return return_tokens
        return False

    def is_address(self, msg):
        msg = msg.strip()
        partition = msg.split(" ")
        if len(partition) < 2:
            return False
        if not partition[0].isdigit():
            return False
        if ' '.join(partition[1:-1]) not in self.domain_info.all_lists[3][1]:
            return False
        else:
            li = msg.split(' ')
            return [li[0]] + [' '.join(li[1:-1])] + [li[-1]]

    def is_name_places(self, msg):
        if not msg[0].isupper():
            return False
        for x in msg.strip().split(" "):
            if x not in self.domain_info.all_lists[5][0]:
                return False
        return msg.split(" ")

    def is_dictionary(self, msg):

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
MAX_TOKENS = 12
NUM_DOMAINS = 7
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
        partial = 0 
        if len(tokens) > MAX_TOKENS:
            partial = 1
            tokens = tokens[:MAX_TOKENS]
        layout = get_layout(len(tokens), domain_id.value)
        word_to_index = self.all_domains.all_dicts[self.all_domains.group_to_lists[domain_id.value]]
        dict_sizes = [len(d) for d in word_to_index]
        try:
            word_indices = [word_to_index[dict_idx][token] for token, dict_idx in zip(tokens, layout)]
        except KeyError:
            print("KeyError")
        max_indices = [dict_sizes[dict_idx] for dict_idx in layout]
        try:
            encoding_len, perm_idx = self.get_encoding_length(word_indices, max_indices)
        except KeyError:
            print("KeyError")
        # print('input perm idx: ', perm_idx)
    
        while encoding_len == -1:
            partial = 1
            word_indices.pop()
            layout.pop()
            max_indices.pop()
            encoding_len, perm_idx = self.get_encoding_length(word_indices, max_indices)
        # print('input word indices: ', word_indices)
        # get the index of the message
        # num = self.tree_index(word_indices, max_indices)
        perm_zero = list(range(52-META_LENGTH-encoding_len, 52-META_LENGTH))
        perm = self.nth_perm(perm_idx, perm_zero)[::-1]
        # print('input perm: ', perm)
        metadata_perm = self.encode_metadata(encoding_len-1, len(tokens)-1, domain_id.value, partial)
        # print('input meta: ', [encoding_len, len(tokens)-1, domain_id.value, partial])
        deck = list(range(0, 52-META_LENGTH-encoding_len)) + perm + metadata_perm

        return deck

    def get_encoding_length(self, indices, max_indices):
        num = self.tree_index(indices, max_indices)
        # print('input actual num: ', num)
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
            n_children = 1
            for factor in max_factors[i+1:]:
                n_children *= factor
            index += factors[i] * n_children
        return index

    # get metadata permutation (6 cards currently)
    def encode_metadata(self, encoding_len, num_tokens, domain_idx, partial):
        if domain_idx == 7:
            domain_idx = 6
        factors = [encoding_len, num_tokens, domain_idx, partial]
        max_factors = [ENCODING_MAX_LENGTH, MAX_TOKENS, NUM_DOMAINS, 2] # partial is just a flag that can only be 0 or 1
        # print('input max factors: ', max_factors)
        meta_idx = self.tree_index(factors, max_factors)
        # print('input meta idx: ', meta_idx)
        meta_perm_zero = list(range(52-META_LENGTH, 52))
        meta_perm = self.nth_perm(meta_idx, meta_perm_zero)
        #print('input perm: ', meta_perm)

        return meta_perm
    
    def nth_perm(self, n, perm_zero):
        perm = []
        items = perm_zero[:]
        factorials = self.factorials_reverse[-len(items):]

        for f in factorials:
            lehmer = n // f
            try:
                x = items.pop(lehmer)
            except IndexError:
                print("ERROR")
            perm.append(x)
            n %= f
        return perm


def smart_format(tokens, template):
    consts = template.split('{}')
    ret = consts[0]
    for token, const in zip(tokens, consts[1:]):
        ret += token + const
    return ret


def assemble_message(tokens, domain_id):
    if domain_id == 0:
        return ''.join(tokens)
    elif domain_id == 1:
        return smart_format(tokens, '{} {}{}{}{} {}')
    elif domain_id == 2:
        return '@' + ''.join(tokens)
    elif domain_id == 3:
        return smart_format(tokens, '{}.{} {}, {}.{} {}')
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
            if 52-META_LENGTH <= card < 52:
                metadata_perm.append(card)
        #print('output perm: ', metadata_perm)
        factors = self.decode_metadata(metadata_perm)
        encoding_len, message_len, domain_id, partial = factors
        encoding_len += 1
        message_len += 1
        # print('output meta: ', [encoding_len, message_len, domain_id, partial])

        message_perm = []
        for card in deck:
            if 52-encoding_len-META_LENGTH <= card < 52-META_LENGTH:
                message_perm.append(card)

        message_perm = message_perm[::-1]
        # print('output perm: ', message_perm)

        if not message_perm:
            message_perm_num = 0
        else:
            message_perm_num = self.perm_number(message_perm)
            # print('output perm number: ', message_perm_num)

        # reconstruct original index from the factorial sum
        actual_num = sum([math.factorial(i) for i in range(encoding_len)]) + message_perm_num
        layout = get_layout(message_len, domain_id)
        if domain_id == 6:
            index_to_word = self.all_domains.all_lists[self.all_domains.group_to_lists[7]]
        else:
            index_to_word = self.all_domains.all_lists[self.all_domains.group_to_lists[domain_id]]
        dict_sizes = [len(d) for d in index_to_word]
        max_factors = [dict_sizes[dict_idx] for dict_idx in layout]
        # print('out actual num: ', actual_num)
        word_indices = self.tree_factors(actual_num, max_factors)
        # print('output word indices: ', word_indices)
        tokens = []
        for word_index, dict_idx in zip(word_indices, layout):
            try:
                tokens.append(index_to_word[dict_idx][word_index])
            except:
                partial = True
                break
        original_message = assemble_message(tokens, domain_id)
        if domain_id == 4:
            original_message += " "
        return 'PARTIAL: ' + original_message if partial else '' + original_message

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
        # print('output max factor: ', max_factors)
        meta_idx = self.perm_number(meta_perm)
        # print('output meta idx: ', meta_idx)
        
        factors = self.tree_factors(meta_idx, max_factors)

        return factors
    
    @staticmethod
    def tree_factors(index, max_factors):
        factors = []
        for i in range(len(max_factors)):
            n_children = 1
            for factor in max_factors[i+1:]:
                n_children *= factor
            factor = index//n_children
            factors.append(factor)
            index %= n_children

        return factors



# Agent====================================================================

class Agent:
    def __init__(self):
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
    msg = 'pneumatoscopes'
    deck = agent.encode(msg)
    msg = agent.decode(deck)
    print('decoded:', msg)
    # msg = '22 West First Street'
    # deck = agent.encode(msg)
    # msg = agent.decode(deck)
    # print('decoded:', msg)


#test_encoder_decoder()


def test_encode_decode_file():
    agent = Agent()
    for i in range(4, 9):
        with open(f'./test_classifier/g{i}_example.txt', 'r') as f:
            msg = f.readline()
            while msg:
                msg = msg.strip()
                deck = agent.encode(msg)
                decode_msg = agent.decode(deck)
                if msg == decode_msg:
                    print("=======================================================")
                    print('pass')
                    print(msg)
                    print(decode_msg)
                else:
                    print("=======================================================")
                    print('failed')
                    print(msg)
                    print(decode_msg)
                msg = f.readline()

#test_encode_decode_file()