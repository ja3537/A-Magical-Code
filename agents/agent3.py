import chunk
import logging
from typing import List, Optional
from cards import valid_deck

from dahuffman import load_shakespeare, HuffmanCodec
from bitstring import Bits
from importlib.metadata import metadata
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
            self.codec = HuffmanCodec.from_frequencies(dictionary)
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
        self.trash_card_start_idx = 32
        self.trash_cards = list(range(self.trash_card_start_idx, 51))
        self.rng = np.random.default_rng(seed=42)
        self.huff = Huffman()  # Create huffman object

        self.max_chunk_size = 6

    def encode(
            self,
            msg: str
        ) -> List[int]:

        bit_str = self.huff.encode(msg, padding_len=0).bin

        step_size, parts, start_padding, end_padding = self.get_parts(bit_str)
        
        if step_size == -1:
            return list(range(52))

        chunk_size = [len(part) for part in parts]
        cards = [int(part, 2) for part in parts]

        if step_size > 0:
            cards = self.hash_msg_with_linear_probe(cards, step_size=step_size)

        print(parts)
        print(cards)
        print(chunk_size)
        print(step_size)

        encode_msg = []

        for i in range(4):
            possible_card = int('0{0:b}'.format(i).zfill(2) + '{0:b}'.format(end_padding).zfill(3), 2)
            if possible_card not in cards:
                encode_msg.append(possible_card)
                break
        else: # No card found
            raise Exception('No card found')
        
        encode_msg.extend(cards)

        #TODO: encode 2 bits for step size - is this needed? or can we just try them all - same thing with padding
        # 3 bits for start padding
        # 3 bits for end padding
        # n bits for each chunk size
        metadata = []
        lengths = ''.join(["0" if size == self.max_chunk_size else "1" for size in chunk_size])
        print(lengths)

        useless_cards = [card for card in range(0, self.trash_card_start_idx)
                         if card not in encode_msg]
        deck = self.trash_cards + useless_cards + metadata + [self.stop_card] + encode_msg

        print("outputs")
        print(valid_deck(deck))
        print(sum(deck) - sum(range(52)))
        print(deck)

        return deck if valid_deck(deck) else list(range(52))
    def decode(
            self,
            deck
    ) -> str:


        deck = self.remove_trash_cards(deck)
        encoded_message = self.get_encoded_message(deck)

        if len(encoded_message) == 0:
            return "NULL"

        last_chunk_len, last_chunk_padding = int('{0:b}'.format(encoded_message[0]).zfill(6)[3:], 2), int('{0:b}'.format(encoded_message[1]).zfill(6)[3:], 2)

        encoded_message = encoded_message[2:]

        bit_str = ''
        seenBitStr = set()
        for idx, card in enumerate(encoded_message[:-1]):
            next_card = encoded_message[idx + 1]
            #currently assume that the next chunnk is max size, MAY NOT BE TRUE THOUGH!!!
            next_bits = '{0:b}'.format(next_card).zfill(self.max_chunk_size)

            normal_chunk = '{0:b}'.format(card).zfill(self.max_chunk_size)
            naked_card = '{0:b}'.format(card)
            minus_one_chunk = naked_card.zfill(self.max_chunk_size-1) + next_bits[:1] if len(naked_card) < self.max_chunk_size else naked_card

            real_chunk_size = self.max_chunk_size

            if minus_one_chunk in seenBitStr:
                real_chunk_size -= 1
                if normal_chunk not in seenBitStr:
                    print("s")
            if normal_chunk in seenBitStr:
                print("ERROR??")
            # else:
            #     real_chunk_size = self.max_chunk_size

            seenBitStr.add('{0:b}'.format(card).zfill(real_chunk_size))
            bit_str += '{0:b}'.format(card).zfill(real_chunk_size)

        last_bits = '{0:b}'.format(encoded_message[-1])[:-last_chunk_padding].zfill(last_chunk_len) if last_chunk_padding > 0 else '{0:b}'.format(encoded_message[-1]).zfill(last_chunk_len)

        bit_str += last_bits

        print(bit_str)
        decoded_message = self.huff.decode(Bits(bin=bit_str), padding_len=0)

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

    def get_parts(self, bit_str):
        '''
        Takes in a bit string 

        Returns a tuple of is_able_to_encode, parts, start_padding, end_padding, contains_no_duplicates
        '''
        chunk_size = self.max_chunk_size
        padding = max(chunk_size, ((chunk_size - len(bit_str) % chunk_size) % chunk_size))
        last_card_padding = 0
    
        for i in range(padding):
            start_padding = i

            padded_bit_str = '0' * start_padding + bit_str
            parts = []
            while len(padded_bit_str) > 0 and chunk_size <= len(padded_bit_str):
                if padded_bit_str[:chunk_size][0] == '0':
                    parts.append(padded_bit_str[:chunk_size])
                    padded_bit_str = padded_bit_str[chunk_size:]
                else:
                    parts.append(padded_bit_str[:chunk_size-1])
                    padded_bit_str = padded_bit_str[chunk_size-1:]

            if len(padded_bit_str) > 0:
                last_card_padding = chunk_size - len(padded_bit_str)
                potential_card = padded_bit_str + '0' * last_card_padding
                if potential_card in parts or potential_card[0] == '1':
                    last_card_padding -= 1
                    potential_card = padded_bit_str + '0' * last_card_padding
                parts.append(potential_card)

            int_deck = [int(part, 2) for part in parts]
            if len(int_deck) == len(set(int_deck)):
                # no duplicates
                return 0, parts, start_padding, last_card_padding

            # must be duplicates, so check if can hash
            canHash, step_size = self.can_hash_msg(int_deck)
            if canHash:
                break
        else:
            return -1, None, None, None
        
        return step_size, parts, start_padding, last_card_padding

    def un_hash_msg(self, encoded_msg, step_size):
        # attempt to unhash the encoded message
        encoded_msg_hash_table = {}

        message = []
        for i, card in enumerate(encoded_msg):
            address_of_card = card
            while address_of_card - step_size in encoded_msg_hash_table:
                address_of_card -= step_size

            encoded_msg_hash_table[address_of_card] = card
            message.append(address_of_card)
        return message

    def hash_msg_with_linear_probe(self, chunks, step_size):
        #attempt a linear probe at the given step size
        hash_table = {}

        encoded_msg = []
        for chunk in chunks:
            address_of_chunk = chunk
            while address_of_chunk in hash_table:
                address_of_chunk += step_size
            hash_table[address_of_chunk] = chunk
            encoded_msg.append(address_of_chunk)
        return encoded_msg

    def can_hash_msg(
            self,
            chunks
    ) -> bool:
        for i in range(1, 5):
            encoded_msg = self.hash_msg_with_linear_probe(chunks, i)
            decoded_hash = self.un_hash_msg(encoded_msg, i)
            if decoded_hash == chunks:
                return True, i
        return False, None

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