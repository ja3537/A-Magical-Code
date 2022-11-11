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


class Domain(Enum):
    ALL = 0
    NUM = 1
    LOWER = 2
    LOWER_AND_UPPER = 3
    LETTERS_NUMBERS = 4
    LAT_LONG = 5


MAX_DOMAIN_VALUE = max([d.value for d in Domain])

DomainFrequencies = {
    # reference of English letter frequencies: https://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
    # password & address
    Domain.ALL: {"a": 8.12, "b": 1.49, "c": 2.71, "d": 4.32, "e": 12.02, "f": 2.30, "g": 2.03, "h": 5.92, "i": 7.31, "j": 0.10, "k": 0.69, "l": 3.98, "m": 2.61, "n": 6.95, "o": 7.68, "p": 1.82, "q": 0.11, "r": 6.02, "s": 6.28, "t": 9.10, "u": 2.88, "v": 1.11, "w": 2.09, "x": 0.17, "y": 2.11, "z": 0.07, " ": 0.11, "\t": 0.10, ".": 6.97, ",": 5.93, "'": 1.53, "\"": 1.33, ":": 0.90, "-": 0.77, ";": 0.74, "?": 0.43, "!": 0.39, "0": 0.09, "1": 0.08, "2": 0.07, "3": 0.06, "4": 0.05, "5": 0.04, "6": 0.03, "7": 0.02, "8": 0.01, "9": 0.005},
    # location
    Domain.LAT_LONG: {"0": 186, "1": 342, "2": 223, "3": 334, "4": 208, "5": 215, "6": 233, "7": 211, "8": 173, "9": 168, "N": 169, "E": 164, "S": 31, "W": 36, ",": 200, ".": 400, " ": 600},
    Domain.LOWER: {"a": 8.12, "b": 1.49, "c": 2.71, "d": 4.32, "e": 12.02, "f": 2.30, "g": 2.03, "h": 5.92, "i": 7.31, "j": 0.10, "k": 0.69, "l": 3.98, "m": 2.61, "n": 6.95, "o": 7.68, "p": 1.82, "q": 0.11, "r": 6.02, "s": 6.28, "t": 9.10, "u": 2.88, "v": 1.11, "w": 2.09, "x": 0.17, "y": 2.11, "z": 0.07, " ": 5},
    # name, places
    Domain.LOWER_AND_UPPER: {"a": 24356, "b": 1881, "c": 3251, "d": 4489, "e": 20854, "f": 919, "g": 2001, "h": 5997, "i": 14284, "j": 271, "k": 2374, "l": 13159, "m": 3469, "n": 15726, "o": 10679, "p": 1148, "q": 400, "r": 12452, "s": 7567, "t": 7377, "u": 4057, "v": 2469, "w": 1482, "x": 266, "y": 4347, "z": 559, "A": 2020, "B": 1534, "C": 2409, "D": 1689, "E": 918, "F": 601, "G": 869, "H": 928, "I": 321, "J": 1621, "K": 1720, "L": 1894, "M": 2221, "N": 865, "O": 468, "P": 841, "Q": 121, "R": 1409, "S": 2600, "T": 1796, "U": 80, "V": 415, "W": 771, "X": 20, "Y": 230, "Z": 175},
    # address(common cases)
    Domain.LETTERS_NUMBERS: {"a": 1594, "b": 67, "c": 181, "d": 768, "e": 2611, "f": 66, "g": 198, "h": 555, "i": 748, "j": 4, "k": 177, "l": 674, "m": 141, "n": 1051, "o": 1265, "p": 79, "q": 6, "r": 1422, "s": 697, "t": 1864, "u": 669, "v": 518, "w": 192, "x": 15, "y": 246, "z": 19, "A": 322, "B": 273, "C": 154, "D": 96, "E": 187, "F": 95, "G": 70, "H": 125, "I": 24, "J": 38, "K": 25, "L": 87, "M": 214, "N": 195, "O": 31, "P": 152, "Q": 5, "R": 383, "S": 566, "T": 69, "U": 14, "V": 27, "W": 261, "X": 3, "Y": 8, "Z": 2, "0": 923, "1": 968, "2": 626, "3": 496, "4": 415, "5": 563, "6": 375, "7": 328, "8": 274, "9": 313, " ": 3577},
    Domain.NUM: {"0": 0.09, "1": 0.08, "2": 0.07, "3": 0.06, "4": 0.05, "5": 0.04, "6": 0.03, "7": 0.02, "8": 0.01, "9": 0.005},
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

    def string_to_binary(self, message: str, domain: Domain) -> str:
        bytes_repr = HuffmanCodec.from_frequencies(
            self.get_domain_frequencies(domain)).encode(message)
        binary_repr = bin(int(bytes_repr.hex(), 16))[2:]
        return binary_repr

    def binary_to_string(self, binary: str, domain: Domain) -> str:
        message_byte = int(binary, 2).to_bytes(
            (int(binary, 2).bit_length() + 7) // 8, 'big')
        message = HuffmanCodec.from_frequencies(
            self.get_domain_frequencies(domain)).decode(message_byte)
        return message

    def deck_encoded(self, message_cards: List[int]) -> List[int]:
        result = []
        for i in range(52):
            if i not in message_cards:
                result.append(i)
        result.extend(message_cards)
        return result

    def get_encoded_cards(self, deck: List[int], start_card_num: int) -> List[int]:
        return [c for c in deck if c >= start_card_num]

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

    def get_hash(self, bit_string: str) -> str:
        hasher = PearsonHasher(1)
        hex_hash = hasher.hash(str(int(bit_string, 2)).encode()).hexdigest()
        return bin(int(hex_hash, 16))[2:].zfill(8)

    def domain_to_binary(self, domain_type: Domain) -> str:
        return bin(int(domain_type.value))[2:].zfill(3)

    def get_domain_type(self, message: str) -> Domain:
        clean_message = "".join(message.split())
        if clean_message.isnumeric():
            return Domain.NUM
        elif clean_message.isalpha() and clean_message.islower():
            return Domain.LOWER
        elif clean_message.isalpha():
            return Domain.LOWER_AND_UPPER
        elif clean_message.isalnum():
            return Domain.LETTERS_NUMBERS
        elif self.is_lat_long(clean_message):
            return Domain.LAT_LONG
        else:
            return Domain.ALL

    def get_domain_frequencies(self, domain: Domain) -> Dict[Domain, Dict[str, float]]:
        return DomainFrequencies[domain] if domain in DomainFrequencies.keys() else DomainFrequencies[Domain.ALL]

    def is_lat_long(self, message: str) -> bool:
        return all([ch.isdigit() or ch in [",", ".", "N", "E", "S", "W"] for ch in message])

    def check_decoded_message(self, message: str, domain_type) -> str:
        clean_message = "".join(message.split())
        if message == '':
            return 'NULL'
        if self.get_domain_type(clean_message) == Domain.ALL:
            if not all(ord(c) < 128 and ord(c) > 32 for c in message):
                return 'NULL'
        return message

    def get_binary_parts(self, binary: str) -> EncodedBinary:
        checksum_bits = binary[-8:]
        domain_bits = binary[-11:-8]
        message_bits = binary[:-11]
        return EncodedBinary(message_bits, domain_bits, checksum_bits)

    def encode(self, message: str) -> List[int]:
        deck = generate_deck(self.rng)
        message = ' '.join(
            [self.word2abrev[word] if word in self.word2abrev else word for word in message.split(" ")])

        domain_type = self.get_domain_type(message)

        binary_repr = self.string_to_binary(message, domain_type)
        binary_repr = binary_repr + \
            self.domain_to_binary(domain_type) + self.get_hash(binary_repr)
        integer_repr = int(binary_repr, 2)

        num_cards_to_encode = 1
        for n in range(1, 52):
            if math.factorial(n) >= integer_repr:
                num_cards_to_encode = n
                break
        message_start_idx = len(deck) - num_cards_to_encode
        message_cards = self.num_to_cards(
            integer_repr, deck[message_start_idx:])
        return self.deck_encoded(message_cards)

    def decode(self, deck: List[int]) -> str:
        message = ''
        domain_type = None
        meet_checksum_count = 0
        for n in reversed(range(1, 51)):
            encoded_cards = self.get_encoded_cards(deck, n)
            integer_repr = self.cards_to_num(encoded_cards)
            binary_repr = bin(int(integer_repr))[2:]
            parts = self.get_binary_parts(binary_repr)
            len_metadata_bits = len(parts.domain_bits) + len(parts.checksum_bits)
            domain_int = int(parts.domain_bits, 2) if parts.domain_bits else MAX_DOMAIN_VALUE + 1

            if len_metadata_bits == 11 and domain_int <= MAX_DOMAIN_VALUE and parts.message_bits and parts.checksum_bits == self.get_hash(parts.message_bits):
                domain_type = Domain(domain_int)
                message = self.binary_to_string(
                    parts.message_bits, domain_type)
                break

        message = self.check_decoded_message(message, domain_type)
        message = ' '.join(
            [self.abrev2word[word] if word in self.abrev2word else word for word in message.split(" ")])

        return message


if __name__ == "__main__":
    agent = Agent()
    message = "Hello"
    deck = agent.encode(message)
    print(deck)
    print(agent.decode(deck))
