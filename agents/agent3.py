import logging
import math
from typing import List, Optional

from dahuffman import load_shakespeare, HuffmanCodec
from bitstring import Bits
import numpy as np


log_level = logging.DEBUG
log_file = 'log/agent3.log'

logger = logging.getLogger('Agent 3')
logger.setLevel(log_level)
logger.addHandler(logging.FileHandler(log_file))


def debug(*args) -> None:
    logger.info(' '.join(args))


# -----------------------------------------------------------------------------
#   Agent Parameters
# -----------------------------------------------------------------------------

CHUNK_SIZE = 5


# -----------------------------------------------------------------------------
#   Codec
# -----------------------------------------------------------------------------

class Huffman:

    def __init__(
            self,
            dictionary: Optional[List[str]] = None,
            chunk_size: int = CHUNK_SIZE
    ) -> None:
        if dictionary is not None:
            self.codec = HuffmanCodec.from_data(''.join(dictionary))
        else:
            self.codec = load_shakespeare()

        self.chunk_size = chunk_size
        self.pad_metadata_size = math.ceil(math.log2(chunk_size -1))

    def _add_padding(self, msg: Bits) -> Bits:
        need_pad_size = self.chunk_size - (len(msg) % self.chunk_size)

        if self.pad_metadata_size > need_pad_size:
            pad_bits_size = need_pad_size + self.chunk_size - self.pad_metadata_size
        else:
            pad_bits_size = need_pad_size - self.pad_metadata_size

        total_pad_size = self.pad_metadata_size + pad_bits_size

        # transform pad_bits_size into binary and add pad_bits_size of 0
        padding = Bits(uint=pad_bits_size << pad_bits_size, length=total_pad_size)
        padded_msg_bin = '0b{}{}'.format(padding.bin, msg.bin)
        padded_msg = Bits(bin=padded_msg_bin)

        debug('[ Huffman._add_padding ]',
                f'len(msg): {len(msg)}, msg: {msg.bin}',
                f'chunk size: {self.chunk_size}',
                f'padding: {padding.bin[:self.pad_metadata_size]}',
                f'+ {padding.bin[self.pad_metadata_size:]}',
                f'padded msg: {padded_msg.bin}')

        return padded_msg

    def _remove_padding(self, msg: Bits) -> Bits:
        pad_bits_size = int(msg.bin[:self.pad_metadata_size], 2)
        msg_start_idx = self.pad_metadata_size + pad_bits_size
        original_encoding = Bits(bin=f'0b{msg.bin[msg_start_idx:]}')

        debug('[ Huffman._remove_padding ]',
                f'padded msg: {msg.bin[:self.pad_metadata_size]}',
                f'+ {msg.bin[self.pad_metadata_size:msg_start_idx]}',
                f'+ {msg.bin[msg_start_idx:]}')

        return original_encoding

    def encode(
            self,
            msg: str
    ) -> Bits:
        bytes = self.codec.encode(msg)
        bits = Bits(bytes=bytes)
        debug('[ Huffman.encode ]', f'msg: {msg} -> bits: {bits.bin}')

        padded_bits = self._add_padding(bits)

        return padded_bits

    def decode(
            self,
            bits: Bits
    ) -> str:
        bits = self._remove_padding(bits)
        decoded = self.codec.decode(bits.tobytes())
        debug('[ Huffman.decode ]', f'bits: {bits.bin} -> msg: {decoded}')

        return decoded


class Agent:
    def __init__(
            self
    ) -> None:
        self.stop_card = 51
        self.trash_cards = list(range(32, 51))
        self.rng = np.random.default_rng(seed=42)
        self.huff = Huffman()  # Create huffman object

    def encode(
            self,
            msg: str
    ) -> List[int]:
        encode_msg = []

        # Convert Huffman to binary
        bin = self.huff.encode(msg).bin

        # Split binary into chunks -> Convert to string to do this
        parts = [str(bin)[i:i+5] for i in range(0, len(str(bin)), 5)]

        # For each binary string convert to binary then int using int()
        # and append to encode_msg
        for i in parts:
            # print(int(Bits(bin=i).bin, 2))
            encode_msg.append(int(Bits(bin=i).bin, 2))

        useless_cards = [card for card in range(0, 32)
                         if card not in encode_msg]
        deck = self.trash_cards + useless_cards + [self.stop_card] + encode_msg

        return deck

    def decode(
            self,
            deck
    ):
        deck = self.remove_trash_cards(deck)
        deck = self.get_encoded_message(deck)
        return "NULL"

    def remove_trash_cards(
            self,
            deck
    ) -> List[int]:
        for i in self.trash_cards:
            deck.remove(i)
        return deck

    def get_encoded_message(
            self,
            deck
    ) -> List[int]:
        deck.index(self.stop_card)
        return deck[deck.index(self.stop_card)+1:]


# -----------------------------------------------------------------------------
#   Unit Tests
# -----------------------------------------------------------------------------

def test_huffman_codec():
    # Note: shakespeare codec doesn't seem to be able to handle punctuations
    cases = ['group 3', 'magic code']

    huffman = Huffman()
    for tc in cases:
        orig = tc
        encoded = huffman.encode(orig)
        decoded = huffman.decode(encoded)

        assert type(
            encoded) == Bits, 'error: encoded message is not of type Bits!'
        assert orig == decoded, 'error: decoded message is not the same as the original'
        assert (len(encoded) % CHUNK_SIZE) == 0, f'error: encoded message has size {len(encoded)}, not a multiple of {CHUNK_SIZE}'

    print('PASSED: Huffman codec using pre-traind shakespeare text')

    cases = ['group 3', 'magic code', 'hi!']
    huffman = Huffman(dictionary=cases)
    for tc in cases:
        orig = tc
        encoded = huffman.encode(orig)
        decoded = huffman.decode(encoded)

        assert type(encoded) == Bits, 'error: encoded message is not of type Bits!'
        assert orig == decoded, 'error: decoded message is not the same as the original'
        assert (len(encoded) % CHUNK_SIZE) == 0, f'error: encoded message has size {len(encoded)}, not a multiple of {CHUNK_SIZE}'

    print('PASSED: Huffman codec using dictionary')
