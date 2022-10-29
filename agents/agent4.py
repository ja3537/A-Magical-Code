from cards import generate_deck
import numpy as np
from typing import List
import math
from pearhash import PearsonHasher
import binascii


class Agent:
    def __init__(self):
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

    def get_domain_type(self, message):
        # 0 --> alphanumeric
        # 1 --> latitude/longitude  i.e: 21 18.41', 157 51.50'
        # 2 --> dates (ignore for now)
        # TODO: rashel - add dates domain

        alphanum = False
        lat_long = False

        for ch in message:
            if ch.isalnum():
                alphanum = True
        if alphanum:
            return '0'

        for ch in message:
            if ord(ch) == 39 or ord(ch) == 44 or ch.isdigit(): # only numbers, commas, apostrophes
                lat_long = True
        if lat_long:
            return '1'

    def encode(self, message):
        deck = generate_deck(self.rng)

        domain_type = self.get_domain_type(message)

        binary_repr = self.string_to_binary(message, domain_type)
        # TODO: rashel - try using 2 bits, not 8
        binary_repr = binary_repr + self.get_hash(binary_repr) + domain_type.zfill(8)
        integer_repr = int(binary_repr, 2)

        num_cards_to_encode = 1
        for n in range(1, 52):
            if math.log2(math.factorial(n)) > len(binary_repr):
                num_cards_to_encode = n
                break
        message_start_idx = len(deck) - num_cards_to_encode
        message_cards = self.num_to_cards(integer_repr, deck[message_start_idx:])

        return self.deck_encoded(message_cards)

    def decode(self, deck):
        message = ''
        for n in reversed(range(1, 51)):
            encoded_cards = self.get_encoded_cards(deck, n)
            integer_repr = self.cards_to_num(encoded_cards)
            binary_repr = bin(int(integer_repr))[2:]
            message_bits = binary_repr[:-16]
            middle_man = binary_repr[:-8]
            hash_bits = middle_man[-8:]
            domain_bits = binary_repr[-8:]
            domain_type = int(domain_bits, 2)
            '''message_bits = binary_repr[:-8]
            hash_bits = binary_repr[-8:]'''

            if len(hash_bits) == 8 and len(message_bits) and hash_bits == self.get_hash(message_bits):
                message = self.binary_to_string(message_bits, domain_type)
                break

        # TODO: rashel - modify to be based on encoding type
        if not all(ord(c) < 128 and ord(c) > 33 for c in message) or message == '':
            return 'NULL'

        return message


if __name__ == "__main__":
    agent = Agent()
    message = "Hello"
    deck = agent.encode(message)
    print(deck)
    print(agent.decode(deck))
