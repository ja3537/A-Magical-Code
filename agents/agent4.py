from cards import generate_deck
import numpy as np
from typing import List
import math


class Agent:
    def __init__(self):
        self.rng = np.random.default_rng(seed=42)
        self.start_marker = 1

    def string_to_binary(self, message):
        return ''.join(format(ord(i), 'b') for i in message)

    def binary_to_string(self, binary):
        return ''.join(chr(int(binary[i * 7:i * 7 + 7], 2)) for i in range(len(binary) // 7))

    def deck_encoded(self, message_cards):
        # message_cards: cards for message
        result = []
        for i in range(52):
            if i != self.start_marker and i not in message_cards:
                result.append(i)

        result.append(self.start_marker)
        result.extend(message_cards)
        return result

    def get_encoded_cards(self, deck):
        marker_indx = deck.index(self.start_marker)
        return [c for c in deck[marker_indx:] if c >= marker_indx]

    def cards_to_num(self, cards: List[int]):
        num_cards = len(cards)

        if num_cards == 1:
            return 0

        ordered_cards = sorted(cards)
        permutations = math.factorial(num_cards)
        sub_list_size = permutations / num_cards
        sub_list_indx = sub_list_size * ordered_cards.index(cards[0])

        return sub_list_indx + self.cards_to_num(cards[1:])

    def num_to_cards(self, num: int, cards: List[int]):
        num_cards = len(cards)

        if num_cards == 1:
            return cards

        ordered_cards = sorted(cards)
        permutations = math.factorial(num_cards)
        sub_list_size = permutations / num_cards
        sub_list_indx = math.floor(float(num) / sub_list_size)
        sub_list_start = sub_list_indx * sub_list_size

        if sub_list_start >= permutations:
            raise Exception('Number too large to encode in cards.')

        first_card = ordered_cards[sub_list_indx]
        ordered_cards.remove(first_card)
        return [first_card, *self.num_to_cards(num - sub_list_start, ordered_cards)]

    def encode(self, message):
        deck = generate_deck(self.rng)

        binary_repr = self.string_to_binary(message)
        integer_repr = int(str(binary_repr), 2)
        message_cards = self.num_to_cards(integer_repr, deck[26:])
        return self.deck_encoded(message_cards)

    def decode(self, deck):
        # return "NULL" if this is a random deck (no message)

        encoded_cards = self.get_encoded_cards(deck)
        integer_repr = self.cards_to_num(encoded_cards)
        binary_repr = bin(int(integer_repr))[2:]
        message = self.binary_to_string(binary_repr)

        return message


if __name__ == "__main__":
    agent = Agent()
    message = "Hello"
    deck = agent.encode(message)
    print(deck)
    print(agent.decode(deck))
