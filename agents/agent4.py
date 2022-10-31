from cards import generate_deck
import numpy as np
from typing import List
import math
from pearhash import PearsonHasher
import binascii
from dahuffman import HuffmanCodec


class Agent:
    def __init__(self):
        self.frequencies = {
            "a": 0.08167,
            "b": 0.01492,
            "c": 0.02782,
            "d": 0.04253,
            "e": 0.12702,
            "f": 0.02228,
            "g": 0.02015,
            "h": 0.06094,
            "i": 0.06966,
            "j": 0.00153,
            "k": 0.00772,
            "l": 0.04025,
            "m": 0.02406,
            "n": 0.06749,
            "o": 0.07507,
            "p": 0.01929,
            "q": 0.00095,
            "r": 0.05987,
            "s": 0.06327,
            "t": 0.09056,
            "u": 0.02758,
            "v": 0.00978,
            "w": 0.02360,
            "x": 0.00150,
            "y": 0.01974,
            "z": 0.00074,
        }
        self.rng = np.random.default_rng(seed=42)

    def string_to_binary(self, message, domain_type):
        return ''.join(format(ord(i), 'b') for i in message)

    def binary_to_string(self, binary, domain_type):
        return ''.join(chr(int(binary[i * 7:i * 7 + 7], 2)) for i in range(len(binary) // 7))

    def deck_encoded(self, message_cards):
        # message_cards: cards for message
        result = []
        for i in range(52):
            if i not in message_cards:
                result.append(i)
        result.extend(message_cards)
        return result

    def get_encoded_cards(self, deck, start_idx):
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

    def domain_to_binary(self, domain_type):
        return bin(int(domain_type))[2:].zfill(2)

    def get_domain_type(self, message):
        message_no_space = "".join(message.split())
        if message.isalnum() or message_no_space.isalnum():
            return '0'
        elif self.is_lat_long(message_no_space):  # ex: 21 18.41', 157 51.50'
            return '1'
        elif self.is_date(message_no_space):
            return '2'
        else:
            return '3'  # do generic encoding

    def is_lat_long(self, message):
        # only numbers, commas, apostrophes
        return all([ch.isdigit() or ch == ',' or ch == "'" for ch in message]) and \
            any(ch.isdigit() for ch in message) and (
                ',' in message) and ("'" in message)

    def is_date(self, message):
        return all([ch.isalnum() or ch == ',' for ch in message])

    def check_decoded_message(self, message, domain_type):
        if message == '':
            return 'NULL'

        message_no_space = "".join(message.split())
        if domain_type == 0:
            if not (message.isalnum() or message_no_space.isalnum()):
                return 'NULL'
        elif domain_type == 1:
            if not self.is_lat_long(message_no_space):
                return 'NULL'
        elif domain_type == 2:
            if not self.is_date(message_no_space):
                return 'NULL'
        elif domain_type == 3:
            if not all(ord(c) < 128 and ord(c) > 32 for c in message):
                return 'NULL'
        return message

    def encode(self, message):
        deck = generate_deck(self.rng)

        domain_type = self.get_domain_type(message)
        domain_binary = self.domain_to_binary(domain_type)
        
        bytes_repr = HuffmanCodec.from_frequencies(
            self.frequencies).encode(message)
        binary_repr = bin(int(bytes_repr.hex(), 16))[2:].zfill(8)
        integer_repr = int.from_bytes(bytes_repr, "big")

        # binary_repr = self.string_to_binary(message, domain_type)
        binary_repr = binary_repr + self.get_hash(binary_repr) + domain_binary
        #print(message, domain_type)
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

    def decode(self, deck):
        message = ''
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
                # message = self.binary_to_string(message_bits, domain_type)
                message_byte = int(message_bits, 2).to_bytes(
                    (int(message_bits, 2).bit_length() + 7) // 8, 'big')
                message = HuffmanCodec.from_frequencies(
                    self.frequencies).decode(message_byte)
                break

        return self.check_decoded_message(message, domain_type)


if __name__ == "__main__":
    agent = Agent()
    message = "Hello"
    deck = agent.encode(message)
    print(deck)
    print(agent.decode(deck))
