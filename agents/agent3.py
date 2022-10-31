import chunk
from concurrent.futures import thread
from curses import meta
import logging
from typing import List, Optional
from cards import valid_deck

from dahuffman import load_shakespeare, HuffmanCodec
from bitstring import Bits
from importlib.metadata import metadata
import numpy as np
import math
from math import factorial as fac
from itertools import groupby


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

class PermutationGenerator:
    # From https://codegolf.stackexchange.com/questions/114883/i-give-you-nth-permutation-you-give-me-n

    def __init__(self):
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.fact = [0] * 34
        self.fact[0] = 1
        for i in range(1, 32):
            self.fact[i] = (self.fact[i - 1] * i)
    def encode(self, cards, rank):
        ''' Encode the given cards into a permutation of the given rank '''
        base = self.alphabet[:len(cards)]

        permutation = self._perm_unrank(rank, base)
        if permutation is None:
            print("trying to create permuation for number ", rank)
            return cards

        return [cards[self.alphabet.index(i)] for i in permutation]

    def decode(self, cards):
        ''' Decode the given permuted cards into a rank '''
        sortedCards = [str(sortedCard) for sortedCard in sorted([int(card) for card in cards])]

        target = ''.join([self.alphabet[sortedCards.index(card)] for card in cards])
        base = self.alphabet[:len(cards)]

        return self._perm_rank(target, base)

    def perm_count(self, s):
        ''' Count the total number of permutations of sorted sequence `s` '''
        n = fac(len(s))
        for _, g in groupby(s):
            n //= fac(sum(1 for u in g))
        return n

    def n_needed(self, messageLength):
        return self.fact[messageLength]

    def _perm_rank(self, target, base):
        ''' Determine the permutation rank of string `target`
            given the rank zero permutation string `base`,
            i.e., the chars in `base` are in lexicographic order.
        '''
        if len(target) < 2:
            return 0
        total = 0
        head, newtarget = target[0], target[1:]
        for i, c in enumerate(base):
            newbase = base[:i] + base[i+1:]
            if c == head:
                return total + self._perm_rank(newtarget, newbase)
            elif i and c == base[i-1]:
                continue
            total += self.perm_count(newbase)
    def _perm_unrank(self, rank, base, head=''):
        ''' Determine the permutation with given rank of the 
            rank zero permutation string `base`.
        '''
        if len(base) < 2:
            return head + ''.join(base)

        total = 0
        for i, c in enumerate(base):
            if i < 1 or c != base[i-1]:
                newbase = base[:i] + base[i+1:]
                newtotal = total + self.perm_count(newbase)
                if newtotal > rank:
                    return self._perm_unrank(rank - total, newbase, head + c)
                total = newtotal

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
        self.permuter = PermutationGenerator()
        self.max_chunk_size = 6

    def encode(
            self,
            msg: str
        ) -> List[int]:

        bit_str = self.huff.encode(msg, padding_len=0).bin

        step_size, parts, start_padding, end_padding = self.get_parts(bit_str)
        
        if step_size == -1:
            # raise Exception("Could not encode message")
            return list(reversed(range(52)))

        chunk_size = [len(part) for part in parts]
        cards = [int(part, 2) for part in parts]

        if step_size > 0:
            cards = self.hash_msg_with_linear_probe(cards, step_size=step_size)

        encode_msg = []

        encode_msg.extend(cards)

        #TODO: encode 2 bits for step size
        # 3 bits for start padding
        # 3 bits for end padding
        # n bits for each chunk size
        lengths = ''.join(["0" if size == self.max_chunk_size else "1" for size in chunk_size])
        end_padding = '{0:b}'.format(end_padding).zfill(3)
        start_padding = '{0:b}'.format(start_padding).zfill(3)
        step_size = '{0:b}'.format(step_size).zfill(2)

        useless_cards = [card for card in range(0, self.trash_card_start_idx)
                         if card not in encode_msg]

        # given the useless_cards, encode the metadata
        metadata_cards = self.encode_metadata(step_size, start_padding, end_padding, lengths, useless_cards)

        useless_cards = [card for card in useless_cards
                    if card not in metadata_cards]

        deck = self.trash_cards + useless_cards + metadata_cards + [self.stop_card] + encode_msg

        return deck if valid_deck(deck) else list(range(52))

    def decode(
            self,
            deck
    ) -> str:
        if deck == list(reversed(range(52))):
            return "Could not encode message"

        deck = self.remove_trash_cards(deck)
        encoded_message = self.get_encoded_message(deck)
        useless_cards = self.get_useless_cards(deck)

        if len(encoded_message) == 0:
            return "NULL"

        messageLength = len(encoded_message)

        # decode metadata
        step_size, start_padding, end_padding, lengths = self.decode_metadata(useless_cards, messageLength)

        if step_size == 0:
            # no linear probing
            cards = encoded_message
        else:
            # linear probing
            cards = self.un_hash_msg(encoded_message, step_size)

        # decode message
        chunk_sizes = [self.max_chunk_size if length == '0' else self.max_chunk_size-1 for length in lengths]
        bit_str = ''.join(['{0:b}'.format(card).zfill(chunk_sizes[i]) for i, card in enumerate(cards)])

        bit_str = bit_str[start_padding:-end_padding] if end_padding > 0 else bit_str[start_padding:]

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
        return deck[deck.index(self.stop_card)+1:]

    def get_useless_cards(self, deck):
        return deck[:deck.index(self.stop_card)]

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
        step_size = step_size if step_size != 3 else 5
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
        step_size = step_size if step_size != 3 else 5

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
        for i in range(1, 4):
            encoded_msg = self.hash_msg_with_linear_probe(chunks, i)
            decoded_hash = self.un_hash_msg(encoded_msg, i)
            if decoded_hash == chunks:
                return True and max(encoded_msg) < 32, i
        return False, None

    def encode_metadata(self, step_size : str, start_padding : str, end_padding : str, lengths : str, cards: List[int]):
        ''' 
        Encode the metadata into the deck

        Returns a list of cards
        '''
        cards = [str(card) for card in cards]

        metadata = step_size + start_padding + end_padding + lengths
        last_n_cards = cards[-self.n_needed_metadata(len(lengths)):]
        permutation = self.permuter.encode(last_n_cards, int(metadata, 2))

        return [int(card) for card in permutation]
        
    def decode_metadata(self, uselessCards : List[int], messageLength : int):
        '''
        Decode the metadata from the deck
        
        Returns a tuple of step_size, start_padding, end_padding, lengths
        '''
        cards = [str(card) for card in uselessCards]
        last_n_cards = cards[-self.n_needed_metadata(messageLength):]
        metadata = self.permuter.decode(last_n_cards)
        metadata = '{0:b}'.format(metadata).zfill(2 + 3 + 3 + messageLength)
        step_size = int(metadata[:2], 2)
        start_padding = int(metadata[2:5], 2)
        end_padding = int(metadata[5:8], 2)
        lengths = metadata[8:8+messageLength]

        return step_size, start_padding, end_padding, lengths

    def n_needed_metadata(self, messageLength : int):
        '''
        Returns the number of cards needed to encode the metadata
        '''
        return self.permuter.n_needed(messageLength)

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