def generate_deck(rng, random=False):
    deck = list(range(52))
    if random:
        rng.shuffle(deck)
    return deck

def valid_deck(deck):
    valid = list(range(52))
    deck.sort()
    return valid == deck