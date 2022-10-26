import random
from itertools import permutations

import numpy as np

import cards


class Agent:
    def __init__(self):
        self.encode_len = 7  # Total items = nPn
        self.valid_cards = list(range(52-self.encode_len, 52))
        self.all_encodings = list(permutations(self.valid_cards))

        # Out of all the encodings generated, keep only a percentage
        # This gives us some guard against NULL decks
        self.percentage_valid = 0.2
        num_valid = int(0.2 * len(self.all_encodings))
        self.valid_encodings = self.all_encodings[:num_valid]

        self.dict_encode = {str(idx): self.valid_encodings[idx] for idx in range(len(self.valid_encodings))}
        self.dict_decode = {self.valid_encodings[idx]: str(idx) for idx in range(len(self.valid_encodings))}
        self.seed = 0
        self.rng = np.random.default_rng(self.seed)

    def encode(self, message):
        if message not in self.dict_encode:
            return ValueError(f"message is not valid. Must be an int less than: {len(self.valid_encodings)}")

        seq_encode = list(self.dict_encode[message])
        # Scramble the rest of the cards above message
        seq_rand = list(range(0, 52-self.encode_len))
        self.rng.shuffle(seq_rand)
        seq_total = seq_rand + seq_encode
        return seq_total

    def decode(self, deck):
        encoding = [c for c in deck if c in self.valid_cards]
        encoding = tuple(encoding)
        if encoding in self.valid_encodings:
            return self.dict_decode[encoding]
        else:
            return "NULL"


if __name__ == "__main__":
    agent = Agent()
    deck = agent.encode("1")
    valid_deck = cards.valid_deck(deck)
    msg = agent.decode(deck)

    # Check NULL redundancy
    count_null = 0
    for idx in range(1000):
        rng = np.random.default_rng(idx)
        deck_r = cards.generate_deck(rng, random=True)
        msg_r = agent.decode(deck_r)
        if msg_r == "NULL":
            count_null += 1
    print("Done")
