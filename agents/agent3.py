from typing import List, Optional

from dahuffman import load_shakespeare, HuffmanCodec
from bitstring import Bits


class Huffman:

    def __init__(self, dictionary: Optional[List[str]] = None) -> None:
        if dictionary is not None:
            self.codec = HuffmanCodec.from_data(''.join(dictionary))
        else:
            self.codec = load_shakespeare()

    def encode(self, msg: str) -> Bits:
        print(f'[ Huffman.encode ] msg: {msg}')
        bytes = self.codec.encode(msg)
        bits = Bits(bytes=bytes)

        return bits

    def decode(self, bits: Bits) -> str:
        return self.codec.decode(bits.tobytes())


class Agent:
    def __init__(self):
        self.stop_card = 51
        self.trash_cards = list(range(32, 51))

    def encode(self, message):
        encoded_message = []

        useless_cards = [card for card in range(0, 32) if card not in encoded_message]
        deck = self.trash_cards + useless_cards + [self.stop_card] + encoded_message
        return deck

    def decode(self, deck):
        deck = self.remove_trash_cards(deck)
        deck = self.get_encoded_message(deck)
        return "NULL"


    def remove_trash_cards(self, deck):
        for i in self.trash_cards:
            deck.remove(i)
        return deck

    def get_encoded_message(self, deck):
        deck.index(self.stop_card)
        return deck[deck.index(self.stop_card)+1:]


# -----------------------------------------------------------------------------
#   Unit Tests
# -----------------------------------------------------------------------------

def test_huffman_codec():
    # Note: the shakespeare codec doesn't seem to be able to handle punctuations
    cases = ['group 3', 'magic code']

    huffman = Huffman()
    for tc in cases:
        orig = tc
        encoded = huffman.encode(orig)
        decoded = huffman.decode(encoded)

        assert type(encoded) == Bits, 'error: encoded message is not of type Bits!'
        assert orig == decoded, 'error: decoded message is not the same as the original'

    print('PASSED: Huffman codec using pre-traind shakespeare text')

    cases = ['group 3', 'magic code', 'hi!']
    huffman = Huffman(dictionary=cases)
    for tc in cases:
        orig = tc
        encoded = huffman.encode(orig)
        decoded = huffman.decode(encoded)

        assert type(encoded) == Bits, 'error: encoded message is not of type Bits!'
        assert orig == decoded, 'error: decoded message is not the same as the original'

    print('PASSED: Huffman codec using dictionary')
