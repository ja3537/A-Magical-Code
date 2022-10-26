from cards import generate_deck
import numpy as np

class Agent:
    def __init__(self):
        self.rng = np.random.default_rng(seed = 42)
        self.stop = 0


    def deck_encoded(self, message_cards):
        # message_cards: cards for message
        result = []
        for i in range(52):
            if i != 1 and i not in message_cards:
                result.append(i)

        result.append(1)
        result.extend(message_cards)
        return result


    def get_encoded_cards(self, deck):
        for i in range(52):
            if deck[i] == 1 and i != (deck.length - 1):
                return deck[i + 1:]
        return []


    def encode(self, message):
        # TODO: Xiaozhou
        # TODO: Noah
        message_cards = [] # noah
        return self.deck_encoded(message_cards)


    def decode(self, deck):
        # return "NULL" if this is a random deck (no message)

        encoded_cards = self.get_encoded_cards(deck)
        # TODO: Noah
        # TODO: Xiaozhou

        return "NULL"