from base64 import encode
import chunk
import logging
import math
from typing import List, Optional
from cards import valid_deck
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

MAX_CHUNK_SIZE = 6


# -----------------------------------------------------------------------------
#   Codec
# -----------------------------------------------------------------------------

class Huffman:

    def __init__(
            self,
            dictionary: Optional[List[str]] = None
    ) -> None:
        if dictionary is not None:
            self.codec = HuffmanCodec.from_data(''.join(dictionary))
        else:
            self.codec = load_shakespeare()

    def _add_padding(self, msg: Bits, padding_len: int) -> Bits:

        padding_bits = '{0:b}'.format(0).zfill(padding_len) if padding_len > 0 else ''
        padded_msg_bin = '0b{}{}'.format(padding_bits, msg.bin)
        padded_msg = Bits(bin=padded_msg_bin)

        debug('[ Huffman._add_padding ]',
                f'len(msg): {len(msg)}, msg: {msg.bin}',
                f'padding size: {padding_len}',
                f'padded msg: {padded_msg.bin}')

        return padded_msg

    def _remove_padding(self, msg: Bits, padding_len: int) -> Bits:
        original_encoding = Bits(bin=f'0b{msg.bin[padding_len:]}')

        debug('[ Huffman._remove_padding ]',
                f'original encoding: {original_encoding.bin}')
        return original_encoding

    def encode(
            self,
            msg: str,
            padding_len: int = 5
    ) -> Bits:
        bytes = self.codec.encode(msg)
        bits = Bits(bytes=bytes)
        debug('[ Huffman.encode ]', f'msg: {msg} -> bits: {bits.bin}')
        padded_bits = self._add_padding(bits, padding_len)

        return padded_bits

    def decode(
            self,
            bits: Bits,
            padding_len: int = 5
    ) -> str:
        bits = self._remove_padding(bits, padding_len)

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
        self.max_chunk_size = MAX_CHUNK_SIZE

    def encode(
            self,
            msg: str
        ) -> List[int]:

        # For each binary string convert to binary then int using int()
        # and then convert list of duplicates to dict to remove duplicates
        # and then back to list

        bit_len = len(self.huff.encode(msg, padding_len=0).bin)

        # Use multiple chunk sizes to find the largest chunk size that allows n duplicate chunks
        for chunk_size in reversed(range(1, self.max_chunk_size + 1)):
            padding =  (chunk_size - (bit_len % chunk_size)) % chunk_size

            # Store metadata of chunk size and padding in first card
            encode_msg = [int('{0:b}'.format(chunk_size).zfill(3) + '{0:b}'.format(padding).zfill(3), 2)]

            # Convert Huffman to binary
            bin = self.huff.encode(msg, padding_len=padding).bin
            
            # Split binary into chunks -> Convert to string to do this
            parts = [str(bin)[i:i+chunk_size] for i in range(0, len(str(bin)), chunk_size)]

            # Convert each chunk to int and add to encode_msg
            for i in parts:
                card = int(Bits(bin=i).bin, 2)
                while card in encode_msg:
                    card += 2**chunk_size
                encode_msg.append(card)

            # Check if encode_msg is valid
            if max(encode_msg) < 32 and len(set(encode_msg)) == len(encode_msg):
                break

        useless_cards = [card for card in range(0, 32)
                         if card not in encode_msg]
        deck = self.trash_cards + useless_cards + [self.stop_card] + encode_msg

        return deck if valid_deck(deck) else list(range(52))

    def decode(
            self,
            deck
    ) -> str:
        deck = self.remove_trash_cards(deck)
        encoded_message = self.get_encoded_message(deck)

        if len(encoded_message) == 0:
            return "NULL"

        metadata = '{0:b}'.format(encoded_message[0]).zfill(6)
        chunk_size, padding = int(metadata[:3], 2), int(metadata[3:], 2)
        encoded_message = encoded_message[1:]

        binString = ''
        for card in encoded_message:
            while card >= 2**chunk_size:
                card -= 2**chunk_size
            binString += '{0:b}'.format(card).zfill(chunk_size)

        decoded_message = self.huff.decode(Bits(bin=binString), padding_len=padding)

        return decoded_message

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