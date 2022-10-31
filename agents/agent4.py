from cards import generate_deck
import numpy as np
from typing import List, Dict
import math
from pearhash import PearsonHasher
from enum import Enum
from dahuffman import HuffmanCodec

class Domain(Enum):
    ALL = 0
    NUM = 1
    LAT_LONG = 2
    DATE = 3

DomainFrequencies = {
    # reference of English letter frequencies: https://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
    Domain.ALL: {"a": 8.12, "b": 1.49, "c": 2.71, "d": 4.32, "e": 12.02, "f": 2.30, "g": 2.03, "h": 5.92, "i": 7.31, "j": 0.10, "k": 0.69, "l": 3.98, "m": 2.61, "n": 6.95, "o": 7.68, "p": 1.82, "q": 0.11, "r": 6.02, "s": 6.28, "t": 9.10, "u": 2.88, "v": 1.11, "w": 2.09, "x": 0.17, "y": 2.11, "z": 0.07, " ": 0.11, "\t": 0.10, ".": 6.97, ",": 5.93, "'": 1.53, "\"": 1.33, ":": 0.90, "-": 0.77, ";": 0.74, "?": 0.43, "!": 0.39, "0": 0.09, "1": 0.08, "2": 0.07, "3": 0.06, "4": 0.05, "5": 0.04, "6": 0.03, "7": 0.02, "8": 0.01, "9": 0.005},
}

class Agent:
    def __init__(self):
        self.rng = np.random.default_rng(seed=42)

    def string_to_binary(self, message: str, domain: Domain) -> str:
        bytes_repr = HuffmanCodec.from_frequencies(self.get_domain_frequencies(domain)).encode(message)
        binary_repr = bin(int(bytes_repr.hex(), 16))[2:].zfill(8)
        return binary_repr

    def binary_to_string(self, binary: str, domain: Domain) -> str:
        message_byte = int(binary, 2).to_bytes((int(binary, 2).bit_length() + 7) // 8, 'big')
        message = HuffmanCodec.from_frequencies(self.get_domain_frequencies(domain)).decode(message_byte)
        return message

    def deck_encoded(self, message_cards: List[int]) -> List[int]:
        # message_cards: cards for message
        result = []
        for i in range(52):
            if i not in message_cards:
                result.append(i)
        result.extend(message_cards)
        return result

    def get_encoded_cards(self, deck: List[int], start_idx: int) -> List[int]:
        return [c for c in deck if c > start_idx]

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
        return bin(int(domain_type.value))[2:].zfill(2)

    def get_domain_type(self, message: str) -> Domain:
        message_no_space = "".join(message.split())
        if message.isalnum() or message_no_space.isalnum():
            return Domain.NUM
        elif self.is_lat_long(message_no_space):  # ex: 21 18.41', 157 51.50'
            return Domain.LAT_LONG
        elif self.is_date(message_no_space):
            return Domain.DATE
        else:
            return Domain.ALL  # do generic encoding
        
    def get_domain_frequencies(self, domain: Domain) -> Dict[Domain, Dict[str, float]]:
        return DomainFrequencies[domain] if domain in DomainFrequencies.keys() else DomainFrequencies[Domain.ALL]

    def is_lat_long(self, message: str) -> bool:
        # only numbers, commas, apostrophes
        return all([ch.isdigit() or ch == ',' or ch == "'" for ch in message]) and \
            any(ch.isdigit() for ch in message) and (
                ',' in message) and ("'" in message)

    def is_date(self, message: str) -> bool:
        return all([ch.isalnum() or ch == ',' for ch in message])

    def check_decoded_message(self, message: str, domain_type: Domain) -> str:
        if message == '':
            return 'NULL'

        message_no_space = "".join(message.split())
        if domain_type == Domain.NUM:
            if not (message.isalnum() or message_no_space.isalnum()):
                return 'NULL'
        elif domain_type == Domain.LAT_LONG:
            if not self.is_lat_long(message_no_space):
                return 'NULL'
        elif domain_type == Domain.DATE:
            if not self.is_date(message_no_space):
                return 'NULL'
        elif domain_type == Domain.ALL:
            if not all(ord(c) < 128 and ord(c) > 32 for c in message):
                return 'NULL'
        return message

    def encode(self, message: str) -> List[int]:
        deck = generate_deck(self.rng)

        domain_type = self.get_domain_type(message)
        domain_binary = self.domain_to_binary(domain_type)

        binary_repr = self.string_to_binary(message, domain_type)
        binary_repr = binary_repr + self.get_hash(binary_repr) + domain_binary
        integer_repr = int(binary_repr, 2)

        num_cards_to_encode = 1
        for n in range(1, 52):
            if math.log2(math.factorial(n)) > len(binary_repr):
                num_cards_to_encode = n
                break
        message_start_idx = len(deck) - num_cards_to_encode
        message_cards = self.num_to_cards(
            integer_repr, deck[message_start_idx:])
        return self.deck_encoded(message_cards)

    def decode(self, deck: List[int]) -> str:
        message = ''
        meet_checksum_count = 0
        for n in reversed(range(1, 51)):
            encoded_cards = self.get_encoded_cards(deck, n)
            integer_repr = self.cards_to_num(encoded_cards)
            binary_repr = bin(int(integer_repr))[2:]
            message_bits = binary_repr[:-10]
            middle_man = binary_repr[:-2]
            hash_bits = middle_man[-8:]
            domain_bits = binary_repr[-2:]
            domain_type = int(domain_bits, 2)

            if len(hash_bits) == 8 and len(message_bits) and hash_bits == self.get_hash(message_bits) and len(domain_bits) == 2:
                message = self.binary_to_string(message_bits, domain_type)

                # TODO: ugly hack to fix the checksum, can be improved
                if meet_checksum_count > 2:
                    break
                meet_checksum_count += 1
                # print(flag, ":" + message)

        return self.check_decoded_message(message, domain_type)


if __name__ == "__main__":
    agent = Agent()
    message = "Hello"
    deck = agent.encode(message)
    print(deck)
    print(agent.decode(deck))
