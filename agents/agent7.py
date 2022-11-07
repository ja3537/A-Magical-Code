from email.headerregistry import Address
from enum import Enum
import math
from collections import defaultdict
import os
import wordninja
import enchant

UNTOK = '*'
EMPTY = ''
DICT_SIZE = 27000
SENTENCE_LEN = 6
ENGLISH_DICTIONARY = enchant.Dict("en_US")

class Domain_Info():
    def __init__(self):
        '''Store all inforamtion about all domains'''
        self.domains = dict()
        self.dictionary = set()
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
        # So far, g1 didnt provide any domain just yet
        self.domains[Domain.G1] = None
    
    def add_g2_domain(self):
        '''Add a new domain to the domain info object'''
        # AIRPORT: abbreviation, location, number
        # Char: upper case, digit, space
        self.domains[Domain.AIRPORT] = []
        with open("./messages/agent2/agent2.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                abbreviation = line.split("\t")[0]
                digits = line.split("\t")[2]
                location = line.strip(abbreviation).strip(digits).strip()
                self.domains[Domain.AIRPORT].append((abbreviation, location, digits))
                line = f.readline()

    def add_g3_domain(self):
        '''Add a new domain to the domain info object'''
        # PASSWORD: @, number/words combination
        # Char: upper/lower case, number, "@"
        self.domains[Domain.PASSWORD] = set()
        with open("./messages/agent3/password_messages.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                line = line.strip("@")
                prev_i = 0
                if line[0].isdigit():
                    sub_str_digit = True
                else:
                    sub_str_digit = False
                msg = []
                for i in range(1, len(line)):
                    if line[i].isdigit() != sub_str_digit:
                        if not line[prev_i:i].isdigit():
                            msg += wordninja.split(line[prev_i:i])
                        prev_i = i
                        sub_str_digit = not sub_str_digit
                for m in msg:  
                    self.domains[Domain.PASSWORD].add(m)
                line = f.readline()
    
    def add_g4_domain(self):
        '''Add a new domain to the domain info object'''
        # Coordinate: latitude, longitude
        # Char: digit, N, E, S, W, ".", ",", space
        self.domains[Domain.LOCATION] = None
    
    def add_g5_domain(self):
        '''Add a new domain to the domain info object'''
        # Address: Address, City, State, Zip
        # Char: words, digit, space, ",", ".", "/"
        words = set()
        with open("./messages/agent5/addresses.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                for i in line.split():
                    if not i.isdigit():
                        words.add(i)
                line = f.readline()
        with open("./messages/agent5/addresses_short.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                for i in line.split():
                    if not i.isdigit():
                        words.add(i)
                line = f.readline()
        self.domains[Domain.ADDRESS] = words
                

    def add_g6_domain(self):
        '''Add a new domain to the domain info object'''
        # N-gram: 1-gram, 2-gram, ..., 9-gram
        # Char: words, digit, space
        self.domains[Domain.NGRAM] = dict()
        for i in range(1, 10):
            self.domains[Domain.NGRAM][i] = set()
            with open("./messages/agent6/corpus-ngram-" + str(i) + ".txt", "r") as f:
                line = f.readline()
                while line:
                    line = line.strip()
                    self.domains[Domain.NGRAM][i].add(line)
                    line = f.readline()

    def add_g7_domain(self):
        '''Add a new domain to the domain info object'''
        # Duh, this is the domain we are working on
        self.domains[Domain.DICTIONARY] = set()
        with open("./messages/agent7/30k.txt", "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                if ENGLISH_DICTIONARY.check(line):
                    self.domains[Domain.DICTIONARY].add(line)
                line = f.readline()
    
    def add_g8_domain(self):
        '''Add a new domain to the domain info object'''
        # Name/Places: Name, places
        # Char: upper/lower case, space
        self.domains[Domain.NAME_PLACES] = dict()
        with open("./messages/agent8/names.txt", "r") as f:
            self.domains[Domain.NAME_PLACES]["names"] = set()
            line = f.readline()
            while line:
                line = line.strip()
                self.domains[Domain.NAME_PLACES]["names"].add(line)
                line = f.readline()
                
        with open("./messages/agent8/places.txt", "r") as f:
            self.domains[Domain.NAME_PLACES]["places"] = set()
            line = f.readline()
            while line:
                line = line.strip()
                self.domains[Domain.NAME_PLACES]["places"].add(line)
                line = f.readline()
    
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
    
class Domain_Classifier():
    def __init__(self) -> None:
        self.domain_info = Domain_Info()

    def is_password(self, msg):
        if msg[0] != "@":
            return False
        THERSHOLD = 0.8
        alphabet_str = ""
        for i in msg:
            if i.isalpha():
                alphabet_str += i
        partition = wordninja.split(alphabet_str)
        is_word = 0
        for i in partition:
            if ENGLISH_DICTIONARY.check(i):
                is_word += 1
        if THERSHOLD < is_word / len(partition):
            return True
        return False
    
    def is_location(self, msg):
        if len(msg.split(",")) != 2:
            return False
        msg = msg.strip()
        c1, c2 = msg.split(",")
        if (c1[-1] == "N" or c1[-1] == "S" ) and (c2[-1] == "E" or c2[-1] == "W"):
            return True
        return False

    def is_airport(self, msg):
        if len(msg.split("\t")) != 3:
            return False
        msg = msg.strip()
        msg = msg.split("\t")
        abbrev = msg[0]
        abbrev_list = [x[0] for x in self.domain_info.get_domain(Domain.AIRPORT)]
        if abbrev in abbrev_list and msg[2].isdigit():
            return True
        return False
    
    def is_dictionary(self, msg):
        if len(msg.split()) > 6 or has_numbers(msg):
            return False
        msg = msg.strip()
        for m in msg.split():
            if m not in self.domain_info.get_domain(Domain.DICTIONARY):
                return False
        return True

    def is_ngram(self, msg):
        msg = msg.strip()
        msg = msg.split()
        for i in range(9, 0, -1):
            if len(msg) % i != 0:
                continue
            i_gram = self.domain_info.get_domain(Domain.NGRAM)[i]
            found_gram = True
            for j in range(0, len(msg), i):
                gram = " ".join(msg[j:j+i])
                if gram not in i_gram:
                    found_gram = False
                    break
            if found_gram:
                return True
        return False

    def is_address(self, msg):
        msg = msg.strip()
        partition = []
        for m in msg.split():
            if not m.isdigit():
                partition.append(m)
        for m in partition:
            if m not in self.domain_info.get_domain(Domain.ADDRESS) \
                or not ENGLISH_DICTIONARY.check(m):
                return False
        return True
    
    def is_name_places(self, msg):
        msg = msg.strip()
        if msg in self.domain_info.get_domain(Domain.NAME_PLACES)["names"]:
            return True
        if msg in self.domain_info.get_domain(Domain.NAME_PLACES)["places"]:
            return True
        return False
    
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
            return Domain.PASSWORD
        elif self.is_location(msg):
            return Domain.LOCATION
        elif self.is_airport(msg):
            return Domain.AIRPORT 
        elif self.is_dictionary(msg):
            return Domain.DICTIONARY
        elif self.is_address(msg):
            return Domain.ADDRESS
        elif self.is_name_places(msg):
            return Domain.NAME_PLACES
        elif self.is_ngram(msg):
            return Domain.NGRAM
        return Domain.G1 # default ascii
            

class EncoderDecoder:
    def __init__(self, n=26):
        self.encoding_len = n
        # characters = " 1234567890abcdefghijklmnopqrstuvwxyz"
        # self.char_dict, self.bin_dict = self.binary_encoding_dicts(characters)
        self.perm_zero = list(range(50-n, 50))
        factorials = [0] * n
        for i in range(n):
            factorials[i] = math.factorial(n-i-1)
        self.factorials = factorials

        words_dict = {EMPTY: 0}
        words_index = [EMPTY]
        with open(os.path.dirname(__file__) + '/../messages/agent7/30k.txt') as f:
            for i in range(1, DICT_SIZE-1):     # reserve 1 for empty and 1 for unknown
                word = f.readline().rstrip('\t\n')
                words_dict[word] = i
                words_index.append(word)
        words_dict[UNTOK] = DICT_SIZE-1
        words_index.append(UNTOK)
        self.words_dict = words_dict
        self.words_index = words_index

    def perm_number(self, permutation):
        n = len(permutation)
        # s = sorted(permutation)
        number = 0

        for i in range(n):
            k = 0
            for j in range(i + 1, n):
                if permutation[j] < permutation[i]:
                    k += 1
            number += k * self.factorials[i]
        return number

    def nth_perm(self, n):
        perm = []
        items = self.perm_zero[:]
        for f in self.factorials:
            lehmer = n // f
            perm.append(items.pop(lehmer))
            n %= f
        return perm

    def str_to_perm(self, message):
        tokens = message.split()
        init = [EMPTY for i in range(SENTENCE_LEN)]
        for i in range(len(tokens)):
            init[i] = tokens[i]
        tokens = init[::-1]
        num = 0
        for i in range(SENTENCE_LEN):
            num += self.words_dict.get(tokens[i], DICT_SIZE-1) * DICT_SIZE**i
        return self.nth_perm(num)

    def perm_to_str(self, perm):
        num = self.perm_number(perm)
        words = []
        for i in range(SENTENCE_LEN):
            index = num % DICT_SIZE
            words.append(self.words_index[index])
            num = num // DICT_SIZE
        return ' '.join(words[::-1]).strip()

class Agent:
    def __init__(self, encoding_len=26):
        self.encoding_len = encoding_len
        self.ed = EncoderDecoder(self.encoding_len)

    def encode(self, message):
        return list(range(50 - self.encoding_len)) + self.ed.str_to_perm(message)[::-1] + [50, 51]

    def decode(self, deck):
        perm = []
        for card in deck:
            if 24 <= card <= 51:
                perm.append(card)
        perm = perm[:-2][::-1] + perm[-2:]

        print(perm)
        if perm[-2:] != [50, 51]:
            return "NULL"
        # if perm[:2] != [22, 23]:
        #     return "PARTIAL:"

        return self.ed.perm_to_str(perm[:-2])



def test_classifier():
    # - G1
    # - AIRPORT
    # - PASSWORD
    # - LOCATION
    # - ADDRESS
    # - NGRAM
    # - DICIONARY
    # - NAME_PLACES
    msg = "@apple123orangekillmenow192"
    msg = "1.1714 S, 36.8356 E"
    msg = "ABL	AMBLER  ALASKA  USA                	132"
    msg = "the of a apple orange kill"
    msg = "6081 Andrews Road Mentor-On-The-Lake Lake OH 44060"
    msg = "New York"
    msg = "sea grain corridor with russia claiming that the corridor is suspended and"
    classifier = Domain_Classifier()
    print(classifier.predict(msg))
    return

test_classifier()