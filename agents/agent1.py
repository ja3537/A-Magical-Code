from itertools import permutations

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
        self.dict_decode = {self.valid_encodings[idx]: idx for idx in range(len(self.valid_encodings))}

    def encode(self, message):
        if message not in self.dict_encode:
            return ValueError(f"message is not valid. Must be an int less than: {len(self.valid_encodings)}")

        return list(range(52))

    def decode(self, deck):
        return "NULL"


if __name__ == "__main__":
    agent = Agent()
    print("Done")
