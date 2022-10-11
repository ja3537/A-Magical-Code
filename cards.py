def generate_deck(rng, random=False):
    deck = list(range(52))
    if random:
        rng.shuffle(deck)
    return deck

def valid_deck(deck):
    if type(deck) != list:
        return False
    valid = list(range(52))
    new_deck = deck.copy()
    new_deck.sort()
    return valid == new_deck