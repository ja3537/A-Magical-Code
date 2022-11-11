from operator import length_hint
from cards import generate_deck
import numpy as np
from typing import List, Dict
import math
from pearhash import PearsonHasher
from enum import Enum
from dahuffman import HuffmanCodec
from collections import namedtuple
import requests
import string
import re


class Domain(Enum):
    ALL = 0                 # Group 1: lowercase letters, period, space, and numbers
    AIRPORT = 1             # Group 2: airport code + random letters/numbers + number
    PASSWORD = 2            # Group 3: @ symbol + random words and numbers
    LAT_LONG = 3            # Group 4: number + N/S + ", " + number + E/W
    STREET = 4              # Group 5: numbers, names, and street suffixes
    WARTIME_NEWS = 5        # Group 6: space delimited english words from wartime correspondences
    SENTENCE = 6            # Group 7: space delimited english words from limited dictionary
    NAME_PLACE = 7          # Group 8: two propper nouns separated by a space


MAX_DOMAIN_VALUE = max([d.value for d in Domain])

DomainFrequencies = {
    # reference of English letter frequencies: https://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
    # Group 1: lowercase letters, period, space, and numbers
    # Domain.ALL: {"a": 8.12, "b": 1.49, "c": 2.71, "d": 4.32, "e": 12.02, "f": 2.30, "g": 2.03, "h": 5.92, "i": 7.31, "j": 0.10, "k": 0.69, "l": 3.98, "m": 2.61, "n": 6.95, "o": 7.68, "p": 1.82, "q": 0.11, "r": 6.02, "s": 6.28, "t": 9.10, "u": 2.88, "v": 1.11, "w": 2.09, "x": 0.17, "y": 2.11, "z": 0.07, " ": 0.11, "\t": 0.10, ".": 6.97, ",": 5.93, "'": 1.53, "\"": 1.33, ":": 0.90, "-": 0.77, ";": 0.74, "?": 0.43, "!": 0.39, "0": 0.09, "1": 0.08, "2": 0.07, "3": 0.06, "4": 0.05, "5": 0.04, "6": 0.03, "7": 0.02, "8": 0.01, "9": 0.005},
    Domain.ALL: {
        "y": 223, "m": 219, "8": 218, "w": 209, "r": 208, "5": 205, "s": 204, "o": 202, "n": 199, "h": 199, "f": 199, "d": 195, "k": 195, "7": 194, "t": 194, "v": 193, "p": 193, "q": 192, "x": 191, "e": 191, "z": 190, "b": 190, "1": 189, "u": 189, "c": 187, "6": 186, "a": 186, "4": 184, "j": 183, "9": 180, "g": 180, "2": 177, "i": 174, "0": 172, "3": 167, " ": 164, "l": 163, ".": 50,
    },
    # Group, 2: random letters/numbers
    Domain.AIRPORT: {
        "1": 137, "Z": 132, "8": 129, "Y": 126, "I": 126, "Q": 126, "6": 125, "W": 124, "T": 124, "R": 122, "2": 119, "F": 119, "P": 116, "M": 114, "O": 113, "3": 112, "D": 112, "E": 110, "U": 109, "G": 107, "7": 105, "B": 105, "N": 105, "L": 105, "C": 105, "X": 105, "4": 104, "J": 104, "5": 101, "9": 100, "A": 97, "K": 96, "S": 95, "V": 95, "H": 89, "0": 87,
    },
    # Group 3: @ symbol + random words and numbers and -
    Domain.PASSWORD: {
        "e": 2314, "i": 1759, "a": 1692, "s": 1529, "r": 1486, "n": 1445, "t": 1436, "o": 1288, "l": 1080, "@": 1000, "c": 869, "d": 795, "p": 629, "u": 595, "m": 567, "g": 560, "1": 426, "h": 425, "6": 416, "2": 407, "9": 386, "4": 381, "7": 375, "8": 374, "5": 373, "y": 368, "3": 359, "b": 346, "0": 338, "f": 279, "v": 267, "w": 187, "k": 168, "x": 71, "z": 61, "j": 45, "q": 33,
    },
    # Group 5: lowercase letters, numbers, space
    Domain.STREET: {
        " ": 2531, "e": 1911, "t": 1358, "a": 1104, "r": 1005, "o": 882, "n": 806, "1": 685, "0": 596, "i": 546, "d": 546, "u": 483, "s": 482, "l": 455, "2": 445, "S": 421, "5": 420, "v": 376, "h": 365, "3": 353, "6": 282, "4": 278, "R": 274, "A": 263, "7": 231, "9": 206, "8": 192, "W": 172, "B": 170, "M": 169, "y": 157, "g": 148, "k": 138, "w": 130, "c": 123, "N": 122, "P": 119, "E": 117, "C": 109, "m": 93, "H": 83, "F": 75, "D": 66, "L": 62, "-": 60, "p": 52, "T": 49, "f": 48, "G": 46, "b": 45, ".": 35, ",": 33, "J": 25, "O": 23, "K": 21, "z": 16, "I": 14, "V": 13, "x": 12, "U": 12, "Y": 8, "Q": 6, "q": 5, "&": 2, "#": 1, "/": 1, "j": 1, "+": 1, "'": 1,
    },
}

DictionaryPaths = {
    Domain.AIRPORT: ['messages/agent2/airportcodes.txt'],
    Domain.PASSWORD: ['messages/agent3/dicts/large_cleaned_long_words.txt'],
    Domain.STREET: ['messages/agent5/street_name.txt', 'messages/agent5/street_suffix.txt'],
    Domain.WARTIME_NEWS: ['messages/agent6/unedited_corpus.txt', 'messages/agent6/corpus-ngram-1.txt', 'messages/agent6/corpus-ngram-2.txt', 'messages/agent6/corpus-ngram-3.txt', 'messages/agent6/corpus-ngram-4.txt', 'messages/agent6/corpus-ngram-5.txt', 'messages/agent6/corpus-ngram-6.txt', 'messages/agent6/corpus-ngram-7.txt', 'messages/agent6/corpus-ngram-8.txt', 'messages/agent6/corpus-ngram-9.txt'],
    Domain.SENTENCE: ['messages/agent3/dicts/30k_cleaned.txt'],
    Domain.NAME_PLACE: ['messages/agent3/dicts/places_and_names.txt']
}

EncodedBinary = namedtuple(
    'EncodedBinary', ['message_bits', 'domain_bits', 'checksum_bits'])


class Agent:
    def __init__(self):
        self.rng = np.random.default_rng(seed=42)
        r = requests.get(
            'https://raw.githubusercontent.com/mwcheng21/minified-text/main/minified.txt')
        minified_text = r.text
        self.abrev2word = {}
        self.word2abrev = {}
        for line in minified_text.splitlines():
            [shortened, full] = line.split(' ')
            self.abrev2word[shortened] = full
            self.word2abrev[full] = shortened

        self.word_to_binary_dicts = {domain: self.get_word_to_binary_dict(
            domain) for domain in Domain if domain in DictionaryPaths.keys()}
        self.binary_to_word_dicts = {domain: self.get_binary_to_word_dict(
            domain) for domain in Domain if domain in DictionaryPaths.keys()}

    # -----------------------------------------------------------------------------
    #   Domain Logic
    # -----------------------------------------------------------------------------
    def get_message_domain(self, message: str) -> Domain:
        matching_domains = []
        words = message.split(' ')

        # Domain.ALL
        if all([ch in list(string.ascii_lowercase + string.digits + '. ') for ch in message]):
            matching_domains.append(Domain.ALL)

        # Domain.AIRPORT
        if (len(words) == 3
            and words[0] in self.word_to_binary_dicts[Domain.AIRPORT].keys()
            and all([ch in list(string.ascii_uppercase + string.digits) for ch in words[1]])
            and all([ch in list(string.digits) for ch in words[2]])
            ):
            matching_domains.append(Domain.AIRPORT)

        # Domain.PASSWORD
        if (message[0] == '@' 
            and all([w in self.word_to_binary_dicts[Domain.PASSWORD].keys() for w in self.get_password_words(message)])
        ):
            matching_domains.append(Domain.PASSWORD)

        # Domain.LAT_LONG
        if all([ch in list('NSEW,. ' + string.digits) for ch in message]):
            matching_domains.append(Domain.LAT_LONG)

        # Domain.STREET
        if (words[0].isnumeric() and
            words[-1] in self.word_to_binary_dicts[Domain.STREET].keys()
            and ' '.join(words[1::-1]) in self.word_to_binary_dicts[Domain.STREET].keys()
        ):
            matching_domains.append(Domain.STREET)
        
        # Domain.WARTIME_NEWS
        if all([word in self.word_to_binary_dicts[Domain.WARTIME_NEWS].keys() for word in words]):
            matching_domains.append(Domain.WARTIME_NEWS)

        # Domain.SENTENCE
        if all([word in self.word_to_binary_dicts[Domain.SENTENCE].keys() for word in words]):
            matching_domains.append(Domain.SENTENCE)

        # Domain.NAME_PLACE
        if all([word in self.word_to_binary_dicts[Domain.NAME_PLACE].keys() for word in words]):
            matching_domains.append(Domain.NAME_PLACE)

        return sorted(matching_domains, key=lambda domain: len(self.message_to_binary(message, domain)))[0]

    def domain_to_binary(self, domain_type: Domain) -> str:
        return bin(int(domain_type.value))[2:].zfill(3)

    def get_domain_frequencies(self, domain: Domain) -> Dict[Domain, Dict[str, float]]:
        return DomainFrequencies[domain] if domain in DomainFrequencies.keys() else DomainFrequencies[Domain.ALL]

    def get_password_words(self, password: str) -> List[str]:
        chunks = [w for w in re.split('|'.join(re.findall(r'\d+', password[1:])), password[1:]) if w]
        words = []
        for chunk in chunks:
            j = 0
            for i in range(len(chunk)+1):
                if chunk[j:i] in self.word_to_binary_dicts[Domain.PASSWORD]:
                    words.append(chunk[j:i])
                    j = i
        return words

    # -----------------------------------------------------------------------------
    #   Message -> Binary & Binary -> Message
    # -----------------------------------------------------------------------------

    def message_to_binary(self, message: str, domain: Domain) -> str:
        if domain == Domain.ALL:
            return self.huff_string_to_binary(message, domain)
        elif domain == Domain.AIRPORT:
            return self.airport_to_binary(message)
        elif domain == Domain.PASSWORD:
            return self.password_to_binary(message)
        elif domain == Domain.LAT_LONG:
            return self.lat_long_to_binary(message)
        elif domain == Domain.STREET:
            return self.street_to_binary(message)
        elif domain == Domain.WARTIME_NEWS:
            return self.wartime_news_to_binary(message)
        elif domain == Domain.SENTENCE:
            return self.sentence_to_binary(message)
        elif domain == Domain.NAME_PLACE:
            return self.name_place_to_binary(message)
        else:
            return self.huff_string_to_binary(message, Domain.ALL)

    def binary_to_message(self, binary: str, domain: Domain) -> str:
        if domain == Domain.ALL:
            return self.huff_binary_to_string(binary, domain)
        elif domain == Domain.AIRPORT:
            return self.binary_to_airport(binary)
        elif domain == Domain.PASSWORD:
            return self.binary_to_password(binary)
        elif domain == Domain.LAT_LONG:
            return self.binary_to_lat_long(binary)
        elif domain == Domain.STREET:
            return self.binary_to_street(binary)
        elif domain == Domain.WARTIME_NEWS:
            return self.binary_to_wartime_news(binary)
        elif domain == Domain.SENTENCE:
            return self.binary_to_sentence(binary)
        elif domain == Domain.NAME_PLACE:
            return self.binary_to_name_place(binary)
        else:
            return self.huff_string_to_binary(binary, Domain.ALL)

    def huff_string_to_binary(self, message: str, domain: Domain) -> str:
        bytes_repr = HuffmanCodec.from_frequencies(
            self.get_domain_frequencies(domain)).encode(message)
        binary_repr = bin(int(bytes_repr.hex(), 16))[2:]
        return binary_repr

    def huff_binary_to_string(self, binary: str, domain: Domain) -> str:
        message_byte = int(binary, 2).to_bytes(
            (int(binary, 2).bit_length() + 7) // 8, 'big')
        message = HuffmanCodec.from_frequencies(
            self.get_domain_frequencies(domain)).decode(message_byte)
        return message

    def get_binary_to_word_dict(self, domain: Domain) -> Dict[str, str]:
        words = []
        for dict_path in DictionaryPaths[domain]:
            with open(dict_path, 'r') as file:
                line = file.readline()
                while line:
                    words.append(line.strip())
                    line = file.readline()
        bits_needed = math.ceil(math.log2(len(words)))
        return {bin(idx)[2:].zfill(bits_needed): word for idx, word in enumerate(words)}

    def get_word_to_binary_dict(self, domain: Domain) -> Dict[str, str]:
        words = []
        for dict_path in DictionaryPaths[domain]:
            with open(dict_path, 'r') as file:
                line = file.readline()
                while line:
                    words.append(line.strip())
                    line = file.readline()
        bits_needed = math.ceil(math.log2(len(words)))
        return {word: bin(idx)[2:].zfill(bits_needed) for idx, word in enumerate(words)}

    def sentence_to_binary(self, message: str) -> str:
        dict = self.word_to_binary_dicts[Domain.SENTENCE]
        return ''.join([dict[word] for word in message.split(' ')])

    def binary_to_sentence(self, binary: str) -> str:
        dict = self.binary_to_word_dicts[Domain.SENTENCE]
        bits_per_word = len(list(dict.keys())[0])
        words_bits = [binary[i:i+bits_per_word]
                      for i in range(0, len(binary), bits_per_word)]
        return ' '.join([dict[bits] for bits in words_bits])

    def name_place_to_binary(self, message: str) -> str:
        dict = self.word_to_binary_dicts[Domain.NAME_PLACE]
        return ''.join([dict[word] for word in message.split(' ')])

    def binary_to_name_place(self, binary: str) -> str:
        dict = self.binary_to_word_dicts[Domain.NAME_PLACE]
        bits_per_word = len(list(dict.keys())[0])
        words_bits = [binary[i:i+bits_per_word]
                      for i in range(0, len(binary), bits_per_word)]
        return ' '.join([dict[bits] for bits in words_bits])

    def airport_code_to_binary(self, airport_code: str) -> str:
        dict = self.word_to_binary_dicts[Domain.AIRPORT]
        return dict[airport_code]

    def binary_to_airport_code(self, binary: str) -> str:
        dict = self.binary_to_word_dicts[Domain.AIRPORT]
        return dict[binary]

    def airport_to_binary(self, message):
        # message: MVM 7PRQ 02202025
        message_list = message.split()
        code1_bin = self.airport_code_to_binary(message_list[0]) # 11 bits zfilled already
        code2_bin = self.huff_string_to_binary(message_list[1], Domain.AIRPORT)
        num_bin = bin(int(message_list[2]))[2:].zfill(24) # max is 12282025, thus 24 bits
        binary_repr = code1_bin + code2_bin + num_bin
        return binary_repr

    def binary_to_airport(self, binary):
        code2_len = len(binary) - 11 - 24

        code1_bin = binary[:11]
        code2_bin = binary[11:11 + code2_len]
        num_bin = binary[-24:]

        code1_str = self.binary_to_airport_code(code1_bin)
        code2_str = self.huff_binary_to_string(code2_bin, Domain.AIRPORT)
        num_str = str(int(num_bin, 2)).zfill(8)

        message = code1_str + " " + code2_str + " " + num_str
        return message
    def password_to_binary(self, message: str) -> str:
        return self.huff_string_to_binary(message[1:], Domain.PASSWORD)

    def binary_to_password(self, binary: str) -> str:
        return '@' + self.huff_binary_to_string(binary, Domain.PASSWORD)

    def lat_long_to_binary(self, message: str) -> str:
        # message: 18.3419 N, 64.9332 W
        # get first number (remove the decimal point), max 7
        # encode next letter with 0 or 1 to represent N or S
        # repeat for next number + E/W
        # pad numbers with zeros on left (once in binary)

        message_list = message.replace(',', '').replace('.', '').split()
        lat_bin = bin(int(message_list[0]))[2:].zfill(20)
        long_bin = bin(int(message_list[2]))[2:].zfill(21)

        lat_dir_bin = '0' if message_list[1] == 'N' else '1'
        long_dir_bin = '0' if message_list[1] == 'E' else '1'

        return lat_bin + lat_dir_bin + long_bin + long_dir_bin

    def binary_to_lat_long(self, binary: str) -> str:
        # RETURN: string in format 18.3419 N, 64.9332 W
        lat_bin = binary[:20]
        lat_dir_bin = binary[20:21]
        long_bin = binary[21:42]
        long_dir_bin = binary[42:]

        lat_num = str(int(lat_bin, 2))
        long_num = str(int(long_bin, 2))
        lat_str = lat_num[:-4] + "." + lat_num[-4:]
        long_str = long_num[:-4] + "." + long_num[-4:]

        lat_dir = 'N' if lat_dir_bin == '0' else 'S'
        long_dir = 'E' if long_dir_bin == '0' else 'W'

        return lat_str + " " + lat_dir + ", " + long_str + " " + long_dir

    def street_to_binary(self, message: str) -> str:
        return self.huff_string_to_binary(message.lower(), Domain.STREET)

    def binary_to_street(self, binary: str) -> str:
        return self.huff_binary_to_string(binary, Domain.STREET).title()

    def wartime_news_to_binary(self, message: str) -> str:
        dict = self.word_to_binary_dicts[Domain.WARTIME_NEWS]
        return dict[message]

    def binary_to_wartime_news(self, binary: str) -> str:
        dict = self.binary_to_word_dicts[Domain.WARTIME_NEWS]
        return dict[binary]

    
    # -----------------------------------------------------------------------------
    #   Binary -> Deck & Deck -> Binary
    # -----------------------------------------------------------------------------

    def cards_to_num(self, cards: List[int]) -> int:
        num_cards = len(cards)

        if num_cards == 1:
            return 0

        ordered_cards = sorted(cards)
        sub_list_size = math.factorial(num_cards - 1)
        sub_list_indx = sub_list_size * ordered_cards.index(cards[0])

        return sub_list_indx + self.cards_to_num(cards[1:])

    def num_to_cards(self, num: int, cards: List[int]) -> List[int]:
        num_cards = len(cards)

        if num_cards == 1:
            return cards

        ordered_cards = sorted(cards)
        permutations = math.factorial(num_cards)
        sub_list_size = math.factorial(num_cards - 1)
        sub_list_indx = math.floor(num / sub_list_size)
        sub_list_start = sub_list_indx * sub_list_size

        if sub_list_start >= permutations:
            raise Exception('Number too large to encode in cards.')

        first_card = ordered_cards[sub_list_indx]
        ordered_cards.remove(first_card)

        return [first_card, *self.num_to_cards(num - sub_list_start, ordered_cards)]


    # -----------------------------------------------------------------------------
    #   Deck Helpers
    # -----------------------------------------------------------------------------

    def get_encoded_deck(self, message_cards: List[int]) -> List[int]:
        result = []
        for i in range(52):
            if i not in message_cards:
                result.append(i)
        result.extend(message_cards)
        return result

    def get_encoded_cards(self, deck: List[int], start_card_num: int) -> List[int]:
        return [c for c in deck if c >= start_card_num]


    # -----------------------------------------------------------------------------
    #   Message Helpers
    # -----------------------------------------------------------------------------

    def get_hash(self, bit_string: str) -> str:
        hasher = PearsonHasher(1)
        hex_hash = hasher.hash(str(int(bit_string, 2)).encode()).hexdigest()
        return bin(int(hex_hash, 16))[2:].zfill(8)

    def check_decoded_message(self, message: str) -> str:
        if message == '':
            return 'NULL'
        try:
            self.get_message_domain(message)
        except:
            return 'NULL'
        return message

    def get_binary_parts(self, binary: str) -> EncodedBinary:
        checksum_bits = binary[-8:]
        domain_bits = binary[-11:-8]
        message_bits = binary[:-11]
        return EncodedBinary(message_bits, domain_bits, checksum_bits)


    # -----------------------------------------------------------------------------
    #   Encode & Decode
    # -----------------------------------------------------------------------------

    def encode(self, message: str) -> List[int]:
        deck = generate_deck(self.rng)
        # message = ' '.join(
        #     [self.word2abrev[word] if word in self.word2abrev else word for word in message.split(" ")])

        domain = self.get_message_domain(message)
        binary_repr = self.message_to_binary(message, domain)
        binary_repr = '1' + binary_repr + \
            self.domain_to_binary(domain) + self.get_hash(binary_repr)
        integer_repr = int(binary_repr, 2)

        num_cards_to_encode = 1
        for n in range(1, 52):
            if math.factorial(n) >= integer_repr:
                num_cards_to_encode = n
                break
        message_start_idx = len(deck) - num_cards_to_encode
        message_cards = self.num_to_cards(
            integer_repr, deck[message_start_idx:])

        return self.get_encoded_deck(message_cards)

    def decode(self, deck: List[int]) -> str:
        message = ''
        domain = None
        for n in range(1, 51):
            encoded_cards = self.get_encoded_cards(deck, n)
            integer_repr = self.cards_to_num(encoded_cards)
            binary_repr = bin(int(integer_repr))[3:]
            parts = self.get_binary_parts(binary_repr)
            domain_int = int(parts.domain_bits, 2) if parts.domain_bits else MAX_DOMAIN_VALUE + 1

            if (len(parts.domain_bits) + len(parts.checksum_bits) == 11 
                and domain_int <= MAX_DOMAIN_VALUE 
                and parts.message_bits 
                and parts.checksum_bits == self.get_hash(parts.message_bits)
            ):
                try:
                    domain = Domain(domain_int)
                    message = self.binary_to_message(parts.message_bits, domain)
                    if domain == self.get_message_domain(message):
                        break
                except:
                    continue
        
        message = self.check_decoded_message(message)
        # message = ' '.join(
        #     [self.abrev2word[word] if word in self.abrev2word else word for word in message.split(" ")])

        return message