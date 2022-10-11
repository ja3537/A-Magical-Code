from cards import generate_deck
import numpy as np

class Agent:
    def __init__(self):
        self.rng = np.random.default_rng(seed = 42)
        self.stop = 0


    def encode(self, message):
        deck = generate_deck(self.rng)
        deck.remove(self.stop)
        encoded_deck = []
        for c in message:
            possible_values = self.char_to_i(c)
            for i in possible_values:
                if i in deck:
                    encoded_deck.append(i)
                    deck.remove(i)
                    break
        encoded_deck.append(self.stop)
        encoded_deck = encoded_deck + deck
        encoded_deck.reverse()
        return encoded_deck

    def decode(self, deck):
        m = ""
        deck.reverse()
        for i in deck:
            if i == self.stop:
                return m
            c = self.i_to_char(i)
            m = m + c


    def char_to_i(self, c):
        value = ord(c) - ord('a') + 1
        return [value, value + 26] #returns all possibilities

    def i_to_char(self, i):
        value = i
        if value > 26:
            value = value - 26
        value = value - 1
        value = ord('a') + value
        return chr(value)