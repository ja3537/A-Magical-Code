from abc import abstractmethod, ABC
from enum import Enum
from itertools import groupby
import logging
import math
from math import factorial as fac
from os.path import isfile
import re
import sys
from typing import List, Optional

from bitstring import Bits
from dahuffman import load_shakespeare, HuffmanCodec
import requests

from cards import valid_deck


# -----------------------------------------------------------------------------
#   Logging
# -----------------------------------------------------------------------------

log_level = logging.INFO
log_file = 'log/agent3.log'

logger = logging.getLogger('Agent 3')
logger.setLevel(log_level)
logger.addHandler(logging.FileHandler(log_file))


def debug(*args) -> None:
    logger.debug(" ".join(map(str, args)))

def info(*args) -> None:
    logger.info(" ".join(map(str, args)))

def error(*args) -> None:
    logger.error("error: " + " ".join(map(str, args)))

if isfile(log_file):
    open(log_file, 'w').close()


# -----------------------------------------------------------------------------
#   Custom Errors
# -----------------------------------------------------------------------------

class NullDeckException(Exception):
    """Raise when a deck doesn't contain a message."""
    pass

def agent_assert(condition: bool, action: Exception):
    """Raise action when the condition/invariant is false/violated."""
    if not condition: raise action


# -----------------------------------------------------------------------------
#   Agent Parameters
# -----------------------------------------------------------------------------

MAX_PERMUTATION_N = 31
MAX_CHUNK_SIZE = 6
NULL_MESSAGE = "NULL"


class PermutationGenerator:
    # From https://codegolf.stackexchange.com/questions/114883/i-give-you-nth-permutation-you-give-me-n

    def __init__(self):
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.fact = [0] * 34
        self.fact[0] = 1
        self.max_fact_n = MAX_PERMUTATION_N
        for i in range(1, self.max_fact_n + 1):
            self.fact[i] = (self.fact[i - 1] * i)

    def _perm_count(self, s: str) -> int:
        ''' Count the total number of permutations of sorted sequence `s` '''
        n = fac(len(s))
        for _, g in groupby(s):
            n //= fac(sum(1 for u in g))
        return n

    def _perm_rank(self, target: str, base: str) -> int:
        ''' Determine the permutation rank of string `target`
            given the rank zero permutation string `base`,
            i.e., the chars in `base` are in lexicographic order.
        '''
        if len(target) < 2:
            return 0
        total = 0
        head, newtarget = target[0], target[1:]
        for i, c in enumerate(base):
            newbase = base[:i] + base[i + 1:]
            if c == head:
                return total + self._perm_rank(newtarget, newbase)
            elif i and c == base[i - 1]:
                continue
            total += self._perm_count(newbase)

    def _perm_unrank(self, rank: int, base: str, head='') -> list[str]:
        ''' Determine the permutation with given rank of the 
            rank zero permutation string `base`.
        '''
        if len(base) < 2:
            return head + ''.join(base)

        total = 0
        for i, c in enumerate(base):
            if i < 1 or c != base[i - 1]:
                newbase = base[:i] + base[i + 1:]
                newtotal = total + self._perm_count(newbase)
                if newtotal > rank:
                    return self._perm_unrank(rank - total, newbase, head + c)
                total = newtotal

    def encode(self, cards: list[str], rank: int) -> list[str]:
        ''' Encode the given cards into a permutation of the given rank '''
        base = self.alphabet[:len(cards)]

        permutation = self._perm_unrank(rank, base)
        if permutation is None:
            error(
                f"trying to create permuation for number {rank} with {len(cards)} cards"
            )
            return cards

        return [cards[self.alphabet.index(i)] for i in permutation]

    def decode(self, cards: list[str]) -> int:
        ''' Decode the given permuted cards into a rank '''
        sortedCards = [
            str(sortedCard)
            for sortedCard in sorted([int(card) for card in cards])
        ]

        target = ''.join(
            [self.alphabet[sortedCards.index(card)] for card in cards])
        base = self.alphabet[:len(cards)]

        return self._perm_rank(target, base)

    def n_needed(self, rank: int) -> int:
        """Returns the number of cards needed to encode a message with
        a bit pattern equal to rank using permutation."""
        for i in range(self.max_fact_n + 1):
            if rank < self.fact[i]:
                return i
        else:
            # return a really large n to indicate to the caller that
            # encoding the rank using permutation won't work
            return sys.maxsize


# -----------------------------------------------------------------------------
#   String -> Domain Detector
# -----------------------------------------------------------------------------

class DomainRule(ABC):
    @abstractmethod
    def verdict(self, msg: str) -> bool:
        pass

class GenericRule:
    def verdict(self, _: str) -> bool:
        return True

class PasswordRule:
    def verdict(self, msg: str) -> bool:
        pattern = re.compile("^@[a-z0-9,.]+$") #TODO: can't handle capitals
        return pattern.match(msg)

class RandomAlphaNumericRule: #not tested
    def verdict(self, msg: str) -> bool:
        pattern = re.compile("^[a-z0-9,.]+$")
        return pattern.match(msg)

class CoordinatesRule:
    def verdict(self, msg: str) -> bool:
        pattern = re.compile("^[0-9]{1,}.[0-9]{4} [NESW], [0-9]{1,}.[0-9]{4} [NESW]$") #TODO: assumes sigfigs of 4
        return pattern.match(msg)

class AddressRule:
    def verdict(self, msg: str) -> bool:
        road_words = ["Street", "Road", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Roadway", "Trail", "Parkway", "Commons", "Av", "Av.", "Ave", "St.", "St", "Rd", "Rd.", "Mall", "Plaza", "Route", "Highway", "Pike", "Pkwy", "Blvd", "Blvd.", "Turnpike", "Expy", "Ct", "Ct.", "Freeway"] #TODO: are there more?
        for word in road_words:
            if word in msg:
                return True
        return False

class SixWordRule: #not tested much
    def verdict(self, msg: str) -> bool:
        # pattern = re.compile("^([a-z]+ ){5}([a-z]+)$") # Changed to accept any num words that are in it 
        pattern = re.compile("^([a-z]+ )*([a-z]+)$")
        with open("./messages/agent7/30k.txt") as f:
            all_words = [word.strip() for word in f.read().splitlines()]
        return pattern.match(msg) and all([word in all_words for word in msg.split(" ")])

class AirplaneFlightRule:
    def verdict(self, msg: str) -> bool:
        striped_msg = ' '.join([word.strip() for word in msg.split()])
        pattern = re.compile("^([A-Z0-9]{3} )([A-Z.'-]* )+([0-9]+)$")#TODO: can't handle lowercase or numbers in middle
        return pattern.match(striped_msg)

class PlaceNameRule: #not tested much
    def verdict(self, msg: str) -> bool:
        pattern = re.compile("^([A-Z][a-z]+) ([A-Z][a-z]+)$")
        with open("./messages/agent8/names.txt") as f: #TODO: find giant list of names and places
            all_words = [word.strip() for word in f.read().splitlines()]
        with open("./messages/agent8/places.txt") as f:
            all_words += [word.strip() for word in f.read().splitlines()]
        return pattern.match(msg) and all([word in all_words for word in msg.split(" ")])

class WarWordsRule:
    def verdict(self, msg: str) -> bool:
        pattern = re.compile("^([a-z]+ )*([a-z]+)$") #TODO: can't handle numbers
        with open("./messages/agent6/corpus-ngram-1.txt") as f: #TODO not sure the actual format
            all_words = [word.strip() for word in f.read().splitlines()]
        return pattern.match(msg) and all([word in all_words for word in msg.split(" ")])


class Domain(Enum):
    GENERIC  = GenericRule
    PASSWORD = PasswordRule
    ALPHA_NUMERIC = RandomAlphaNumericRule
    COORDS = CoordinatesRule
    ADDRESS = AddressRule
    SIX_WORDS = SixWordRule
    FLIGHTS = AirplaneFlightRule
    PLACES_AND_NAMES = PlaceNameRule
    WAR_WORDS = WarWordsRule

class DomainDetector:
    def __init__(self, domains: list[Domain], default_domain=Domain.GENERIC):
        self.domains = domains
        self.default_domain = default_domain

    def detect(self, msg: str) -> Domain:
        for domain in self.domains:
            rule = domain.value()
            if rule.verdict(msg):
                return domain
        return Domain.GENERIC


# -----------------------------------------------------------------------------
#   String <-> String Message TransFormer (Compression, DeCompression)
# -----------------------------------------------------------------------------

class MessageTransformer(ABC):
    @abstractmethod
    def compress(self, msg: str) -> tuple[str, Bits]:
        pass

    @abstractmethod
    def uncompress(self, bits: Bits) -> str:
        pass


class GenericTransformer(MessageTransformer):
    def __init__(self):
        self.huffman = Huffman()

    def compress(self, msg: str) -> tuple[str, Bits]:
        bits = self.huffman.encode(msg, padding_len=0)
        debug(f'GenericTransformer: "{msg}" -> {bits.bin}')

        return msg, bits

    def uncompress(self, bits: Bits) -> str:
        msg = self.huffman.decode(bits, padding_len=0)
        debug(f'GenericTransformer: "{bits.bin}" -> {msg}')

        return msg

    @classmethod
    def __str__(cls) -> str:
        return "GenericTransformer"

class DictIdxTransformer(MessageTransformer):
    def __init__(self, wordlist_path: str, delimiter=" "):
        self.delimiter = delimiter
        with open(wordlist_path) as f:
            self.wordlist = [word.strip() for word in f.read().splitlines()]
        self.wordlist.append("*")
        self.word2idx = {word: idx for idx, word in enumerate(self.wordlist)}
        self.idx2word = {idx: word for idx, word in enumerate(self.wordlist)}
        self.word_bit_size = int(math.ceil(math.log2(len(self.wordlist))))

    def compress(self, msg: str) -> tuple[str, Bits]:
        words = msg.split(self.delimiter)
        encoded = []
        for word in words:
            if word not in self.word2idx:
                word = "*"
            idx = self.word2idx[word]
            encoded.append("{0:b}".format(idx).zfill(self.word_bit_size))
        bits = Bits(bin="".join(encoded))
        
        debug(f'DictIdxTransformer: "{msg}" -> {bits.bin}')

        return msg, bits

    def uncompress(self, bits: Bits) -> str:
        words = []
        for i in range(0, len(bits.bin), self.word_bit_size):
            idx = int(bits.bin[i:i+self.word_bit_size], 2)
            words.append(self.idx2word[idx])
        msg = self.delimiter.join(words)
        
        debug(f'DictIdxTransformer: "{bits.bin}" -> {msg}')

        return msg

class WordTransformer(MessageTransformer):
    def __init__(self, delimiter=" ", wordlist=None, huffman_encoding=None):
        self.huffman = Huffman() if huffman_encoding is None else huffman_encoding

        #---------------------------------------------------------------------
        #  Obtain abbreviation for words in a dictionary.
        #---------------------------------------------------------------------
        # Note: self.word2abrev stores the mapping of word to abbreviation
        self.delim = delimiter

        # if can use precomp, comment this out
        if wordlist is None:
            r = requests.get(
                'https://raw.githubusercontent.com/mwcheng21/minified-text/main/minified.txt'
            )
            minified_text = r.text
            self.abrev2word = {}
            self.word2abrev = {}
            for line in minified_text.splitlines():
                [shortened, full] = line.split(' ')
                self.abrev2word[shortened] = full
                self.word2abrev[full] = shortened
        else:
            with open(wordlist, 'r') as f:
                self.abrev2word = {}
                self.word2abrev = {}
                for line in f.readlines():
                    line = line.strip()
                    [shortened, full] = line.split(' ')
                    self.abrev2word[shortened] = full
                    self.word2abrev[full] = shortened

    def compress(self, msg: str) -> tuple[str,Bits]:
        msg = self.delim.join([
            self.word2abrev[word] if word in self.word2abrev else word
            for word in msg.split(self.delim)
        ])
        bits = self.huffman.encode(msg, padding_len=0)

        return msg, bits

    def uncompress(self, bits: Bits) -> str:
        decoded_message = self.huffman.decode(bits, padding_len=0)
        original_message = self.delim.join([
            self.abrev2word[word] if word in self.abrev2word else word
            for word in decoded_message.split(self.delim)
        ])

        return original_message

    @classmethod
    def __str__(cls) -> str:
        return "WordTransformer"

class PasswordsTransformer(MessageTransformer):
    def __init__(self):
        self.freq = {'1': 10000, '2': 10000, '3': 10000, '4': 10000, '5': 10000, '6': 10000, '7': 10000, '8': 10000, '9': 10000, '0': 10000, 'e': 12351, 'a': 10065, 's': 9847, 'r': 8604, 'i': 8586, 't': 7560, 'n': 7484, 'o': 7095, 'c': 5985, 'l': 5970, 'd': 5057, 'p': 4221, 'm': 4148, 'u': 3405, 'g': 3281, 'h': 3040, 'b': 2787, 'f': 2164, 'y': 1891, 'v': 1602, 'w': 1579, 'k': 1390, 'x': 532, 'j': 486, 'z': 371, 'q': 284} #TODO: change numerical freqs to be more realistic
        self.huffman = Huffman(self.freq)
        with open("messages/agent3/dicts/shortened_dicts/passwords_mini.txt", 'r') as f:
            self.abrev2word = {}
            self.word2abrev = {}
            for line in f.readlines():
                line = line.strip()
                [shortened, full] = line.split(' ')
                self.abrev2word[shortened] = full
                self.word2abrev[full] = shortened

    def compress(self, msg: str) -> tuple[str, Bits]:
        msg = msg.replace("@", "")
        splitMsg = self._get_all_words(msg, self.word2abrev.keys())

        combinedMsg = ''.join([
            self.word2abrev[word] if word.isalpha() and word in self.word2abrev else word
            for word in splitMsg
        ])

        #TODO: join on space or not?
        bits = self.huffman.encode(combinedMsg, padding_len=0)

        return combinedMsg, bits

    def uncompress(self, bits: Bits) -> str:
        msg = self.huffman.decode(bits, padding_len=0)
        splitMsg = self._get_all_words(msg, self.abrev2word.keys())

        combinedMsg = '@' + ''.join([
            self.abrev2word[word] if word.isalpha() and word in self.abrev2word else word 
            for word in splitMsg
        ])

        return combinedMsg

    def _get_all_words(self, msg: str, wordlist) -> list[str]:
        words = []
        startIdx = 0
        for i in range(1, len(msg)):
            if msg[i].isalpha() != msg[i-1].isalpha():
                words.append(msg[startIdx:i])
                startIdx = i
        words.append(msg[startIdx:])

        splitMsg = []
        for word in words:
            if word.isalpha():
                extractedWords = list(reversed(self._get_word_helper(wordlist, [], word)))

                if len(extractedWords) == 0:
                    splitMsg.append(word)
                else:
                    splitMsg.extend(extractedWords)
            else:
                splitMsg.append(word)
        return splitMsg

    def _get_word_helper(self, wordlist, words, current_string):
        '''
        Recursive helper function for _get_all_words

        Returns a list of words that can be extracted from the current_string
        (ie current_string = "hello" and wordlist = ["he", "llo"] -> ["he", "llo"])

        #doesn't work for cases like "searchare" -> ['se', 'arc', 'hare'] and in abrev ['sear', 'chare'] instead of ['search', 'are']
        However, we assume that this never happens, or rarely if it does
        '''
        if len(current_string) == 0:
            return words
        if current_string in wordlist:
            words.append(current_string)
            return words
        
        for i in range(len(current_string)):
            currWord = current_string[i:]
            if currWord in wordlist:
                words.append(currWord)
                return self._get_word_helper(wordlist, words, current_string[:i])

        return words

    @classmethod
    def __str__(cls) -> str:
        return "PasswordTransformer"

class CoordsTransformer(MessageTransformer):
    def __init__(self):
        self.huffman = Huffman()

    def compress(self, msg: str) -> tuple[str, Bits]:
        # 0-180 0000-9999 0-180 0000-9999 0-16
        info = re.split('[ .]', msg.replace(",", ""))
        lat, latMin, latDir, long, longMin, longDir = info
        for i in range(2, 9):
            try:
                bits = self._intstr_to_bitstr(lat, i) + self._intstr_to_bitstr(latMin, i+6)
                bits += self._intstr_to_bitstr(long, i) + self._intstr_to_bitstr(longMin, i+6)
                bits += ("0" if latDir == "N" else "1") + ("0" if longDir == "E" else "1")
                break
            except ValueError:
                pass

        #TODO: can I decrease the n bits for each number even more?
        return msg, Bits(bin=bits)

    def uncompress(self, bits: Bits) -> str:
        bitstr = bits.bin
        if len(bitstr) - 14 < 8 or (len(bitstr) -14) % 4 != 0:
            raise NullDeckException("NULL(Weird, look into, happens 4 times)")

        i = int((len(bitstr) - 14) / 4)

        lat = self._bitstr_to_intstr(bitstr[:i])
        latMin = self._bitstr_to_intstr(bitstr[i:i+i+6], padding=4)
        long = self._bitstr_to_intstr(bitstr[i+i+6:i+i+6+i])
        longMin = self._bitstr_to_intstr(bitstr[i+i+6+i:i+i+6+i+i+6], padding=4)
        latDir = "N" if bitstr[i+i+6+i+i+6] == "0" else "S"
        longDir = "E" if bitstr[i+i+6+i+i+6+1] == "0" else "W"

        formatedOutput = f"{lat}.{latMin} {latDir}, {long}.{longMin} {longDir}"

        return formatedOutput

    def _intstr_to_bitstr(self, num: str, num_bits: int) -> Bits:
        return Bits(uint=int(num), length=num_bits).bin
  
    def _bitstr_to_intstr(self, bits: str, padding=0) -> str:
        num = str(int(bits, 2))
        if padding:
            return "0"*(4 - len(num)) + num
        return num

    @classmethod
    def __str__(cls):
        return "CoordsTransformer"

class AddressTransformer(MessageTransformer): #TODO: later bc format not finalized
    def __init__(self):
        self.huffman = Huffman()

    def compress(self, msg: str) -> tuple[str, Bits]:
        bits = self.huffman.encode(msg, padding_len=0)
        debug(f'GenericTransformer: "{msg}" -> {bits.bin}')

        return msg, bits

    def uncompress(self, bits: Bits) -> str:
        msg = self.huffman.decode(bits, padding_len=0)
        debug(f'GenericTransformer: "{bits.bin}" -> {msg}')

        return msg

    @classmethod
    def __str__(cls) -> str:
        return "AddressTransformer"
       
class FlightsTransformer(MessageTransformer): #TODO: later bc format not finalized
    def __init__(self):
        self.huffman = Huffman()

    def compress(self, msg: str) -> tuple[str, Bits]:
        bits = self.huffman.encode(msg, padding_len=0)
        debug(f'GenericTransformer: "{msg}" -> {bits.bin}')

        return msg, bits

    def uncompress(self, bits: Bits) -> str:
        msg = self.huffman.decode(bits, padding_len=0)
        debug(f'GenericTransformer: "{bits.bin}" -> {msg}')

        return msg

    @classmethod
    def __str__(cls) -> str:
        return "FlightsTransformer"

# TODO: adding freq prevents cases of things not in it
class WarWordsTransformer(MessageTransformer):
    def __init__(self):
        self.freq = {' ': 15000, 'e': 2490, 's': 1960, 'a': 1852, 'r': 1672, 'i': 1648, 't': 1488, 'n': 1482, 'o': 1307, 'd': 1168, 'l': 1147, 'c': 1106, 'p': 835, 'm': 727, 'g': 668, 'u': 645, 'h': 542, 'b': 483, 'f': 468, 'v': 367, 'w': 348, 'k': 322, 'y': 312, '0': 140, '1': 96, '2': 77, 'j': 68, 'z': 58, '5': 55, 'x': 52, '3': 50, '4': 45, '7': 34, '9': 34, 'q': 30, '8': 29, '6': 25}
        self.huffman = Huffman(self.freq)
        self.word_file = "./messages/agent3/dicts/shortened_dicts/war_words_mini.txt"
        self.transformer = WordTransformer(wordlist=self.word_file, huffman_encoding=self.huffman)

    def compress(self, msg: str) -> tuple[str, Bits]: #TODO: maybe clean messages?
        return self.transformer.compress(msg)

    def uncompress(self, bits: Bits) -> str:
        return self.transformer.uncompress(bits)

    @classmethod
    def __str__(cls) -> str:
        return "WarWordsTransformer"

class PlacesAndNamesTransformer(MessageTransformer):
    def __init__(self):
        self.freq = {' ': 25000, 'a': 15393, 'e': 11709, 'n': 9319, 'l': 8288, 'r': 7848, 'i': 7821, 's': 6479, 'o': 5599, 't': 5435, 'h': 4205, 'm': 4110, 'd': 3968, 'c': 3925, 'k': 3022, 'y': 2902, 'b': 2377, 'u': 2122, 'g': 1805, 'j': 1767, 'v': 1601, 'w': 1476, 'p': 1375, 'f': 1041, 'z': 518, 'q': 323, 'x': 169}
        self.huffman = Huffman(self.freq)
        self.word_file = "./messages/agent3/dicts/shortened_dicts/places_and_names_mini.txt"
        self.transformer = WordTransformer(wordlist=self.word_file, huffman_encoding=self.huffman)

    def compress(self, msg: str) -> tuple[str, Bits]:
        return self.transformer.compress(msg.lower())

    def uncompress(self, bits: Bits) -> str:
        decoded_msg = self.transformer.uncompress(bits)
        capitalized_msg = [word.capitalize() for word in decoded_msg.split()]
        return " ".join(capitalized_msg)

    @classmethod
    def __str__(cls) -> str:
        return "PlacesAndNamesTransformer"

class SixWordsTransformer(MessageTransformer):
    def __init__(self):
        self.word_file = "./messages/agent3/dicts/six_words_list.txt"
        self.transformer = DictIdxTransformer(wordlist_path=self.word_file)

    def compress(self, msg: str) -> tuple[str, Bits]:
        return self.transformer.compress(msg)

    def uncompress(self, bits: Bits) -> str:
        return self.transformer.uncompress(bits)

    @classmethod
    def __str__(cls) -> str:
        return "SixWordsTransformer"

class AlphaNumericTransformer(MessageTransformer):
    def __init__(self):
        self.freq = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1, "g": 1, "h": 1, "i": 1, "j": 1, "k": 1, "l": 1, "m": 1, "n": 1, "o": 1, "p": 1, "q": 1, "r": 1, "s": 1, "t": 1, "u": 1, "v": 1, "w": 1, "x": 1, "y": 1, "z": 1, "0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, ".": 1, ",": 1, " ": 1}
        self.huffman = Huffman(self.freq)

    def compress(self, msg: str) -> tuple[str, Bits]:
        bits = self.huffman.encode(msg, padding_len=0)
        debug(f'GenericTransformer: "{msg}" -> {bits.bin}')

        return msg, bits

    def uncompress(self, bits: Bits) -> str:
        msg = self.huffman.decode(bits, padding_len=0)
        debug(f'GenericTransformer: "{bits.bin}" -> {msg}')

        return msg

    @classmethod
    def __str__(cls) -> str:
        return "AlphaNumericTransformer"
       

# -----------------------------------------------------------------------------
#   Bits <-> Deck Converter (BDC)
# -----------------------------------------------------------------------------
Deck = list[int]

class BDC(ABC):
    """Bits <-> Deck Converter."""
    def __init__(self, cards: Deck):
        self._cards = cards # cards that can be used to encode bits

    @property
    def cards(self):
        return self._cards

    @abstractmethod
    def to_deck(self, bits: Bits) -> Optional[tuple[Deck, Deck]]:
        """Returns message deck, including any message metadata."""
        pass

    @abstractmethod
    def to_bits(self, msg: Deck, msg_metadata: Deck) -> Optional[Bits]:
        pass

class ChunkConverter(BDC):
    """Converts between bits and deck of cards by mapping each chunk of bits
    to a card, using linear probing if there are collisions."""

    def __init__(self, cards: Deck, max_chunk_size=MAX_CHUNK_SIZE):
        super().__init__(cards)
        self.permuter = PermutationGenerator()
        self.trash_card_start_idx = 32
        self.max_chunk_size = max_chunk_size

    def _n_needed_metadata(self, messageLength: int):
        """Returns the number of cards needed to encode the metadata."""
        return self.permuter.n_needed(messageLength)

    def _encode_metadata(self, step_size: str, start_padding: str,
                        end_padding: str, lengths: str, cards: Deck
                        ) -> Deck:
        """Encode the message metadata into a deck.""" 
        cards = [str(card) for card in cards]

        metadata = step_size + start_padding + end_padding + lengths

        last_n_cards = cards[-self._n_needed_metadata(2**len(metadata)):]
        permutation = self.permuter.encode(last_n_cards, int(metadata, 2))

        return [int(card) for card in permutation]

    def _decode_metadata(self, msg_metadata: Deck, msg_len: int):
        """Decodes the metadata from the deck.
        
        Returns a tuple of step_size, start_padding, end_padding, lengths.
        """
        cards = [str(card) for card in msg_metadata]
        last_n_cards = cards[-self._n_needed_metadata(2**(msg_len + 3 + 3 + 2)):]

        metadata = self.permuter.decode(last_n_cards)
        metadata = '{0:b}'.format(metadata).zfill(2 + 3 + 3 + msg_len)
        step_size = int(metadata[:2], 2)
        start_padding = int(metadata[2:5], 2)
        end_padding = int(metadata[5:8], 2)
        lengths = metadata[8:8 + msg_len]

        return step_size, start_padding, end_padding, lengths

    def _unhash_msg(self, encoded_msg, step_size):
        """Attempts to unhash the encoded message."""
        step_size = step_size if step_size != 3 else 5
        encoded_msg_hash_table = {}

        message = []
        for i, card in enumerate(encoded_msg):
            address_of_card = card
            while address_of_card - step_size in encoded_msg_hash_table:
                address_of_card -= step_size

            encoded_msg_hash_table[card] = address_of_card
            message.append(address_of_card)
        return message

    def _hash_msg_with_linear_probe(self, chunks, step_size):
        """Attempts a linear probe at the given step size."""
        hash_table = {}
        step_size = step_size if step_size != 3 else 5

        encoded_msg = []
        for chunk in chunks:
            address_of_chunk = chunk
            # TODO: can't know if [26, 27] corresponds to [26, 27] or [26, 26], without
            # metadata
            while address_of_chunk in hash_table or address_of_chunk not in self.cards:
                address_of_chunk += step_size

                # if address_of_chunk is larger than the largest card we can use,
                # we can't use it to encode our message, thus return an empty array
                # to indicate that we can't encode using linear probing
                #
                # TODO: use mod, so that after address_of_chunk is greater than the
                # maximum, wrap around.
                if address_of_chunk > max(self.cards):
                    return []

            hash_table[address_of_chunk] = chunk
            encoded_msg.append(address_of_chunk)

        return encoded_msg

    def _can_hash_msg(self, chunks) -> bool:
        for i in range(1, 4):
            encoded_msg = self._hash_msg_with_linear_probe(chunks, i)
            decoded_hash = self._unhash_msg(encoded_msg, i)
            if decoded_hash == chunks:
                return True and max(encoded_msg) < 32, i
        return False, None

    def _get_parts(self, bit_str):
        """Takes in a bit string, checks if it's possible to encode.
        
        Returns a tuple of is_able_to_encode, parts, start_padding,
                           end_padding, contains_no_duplicates.
        """
        padding = chunk_size = self.max_chunk_size
        last_card_padding = 0

        for i in range(padding):
            start_padding = i

            padded_bit_str = '0' * start_padding + bit_str
            parts = []
            while len(padded_bit_str) > 0 and chunk_size <= len(
                    padded_bit_str):
                if padded_bit_str[:chunk_size][0] == '0':
                    parts.append(padded_bit_str[:chunk_size])
                    padded_bit_str = padded_bit_str[chunk_size:]
                else:
                    parts.append(padded_bit_str[:chunk_size - 1])
                    padded_bit_str = padded_bit_str[chunk_size - 1:]

            if len(padded_bit_str) > 0:
                last_card_padding = chunk_size - len(padded_bit_str)
                potential_card = padded_bit_str + '0' * last_card_padding
                if potential_card in parts or potential_card[0] == '1':
                    last_card_padding -= 1
                    potential_card = padded_bit_str + '0' * last_card_padding
                parts.append(potential_card)

            int_deck = [int(part, 2) for part in parts]
            _int_deck_set = set(int_deck)
            if (len(int_deck) == len(_int_deck_set)
                and _int_deck_set.issubset(set(self.cards))):
                # no duplicates and only uses available cards
                return 0, parts, start_padding, last_card_padding

            # must be duplicates, so check if can hash
            canHash, step_size = self._can_hash_msg(int_deck)
            if canHash:
                break
        else:
            return -1, None, None, None

        return step_size, parts, start_padding, last_card_padding

    def to_deck(self, bits: Bits) -> Optional[tuple[Deck, Deck]]:
        step_size, parts, start_padding, end_padding = self._get_parts(bits.bin)

        if step_size < 0:
            # could not encode message
            return None

        chunk_size = [len(part) for part in parts]
        msg_cards = [int(part, 2) for part in parts]

        if step_size > 0:
            # has duplicate bit chunks, hash with linear probing
            msg_cards = self._hash_msg_with_linear_probe(msg_cards, step_size=step_size)

        # Message Metadata Format:
        #       + --------- + ------------- + ----------- + --------------------- +
        #       | step size | start padding | end padding | chunk size of n cards |
        #       + --------- + ------------- + ----------- + --------------------- +
        # bits:       2             3              3                 n
        lengths = "".join([
            "0" if size == self.max_chunk_size else "1"
            for size in chunk_size
        ])
        end_padding = '{0:b}'.format(end_padding).zfill(3)
        start_padding = '{0:b}'.format(start_padding).zfill(3)
        step_size = '{0:b}'.format(step_size).zfill(2)

        # given the unused cards, encode the metadata
        unused_cards = [
            card for card in self.cards
            if card not in msg_cards
        ]
        msg_metadata_cards = self._encode_metadata(
            step_size, start_padding, end_padding, lengths, unused_cards)

        return msg_metadata_cards, msg_cards

    def to_bits(self, msg: Deck, msg_metadata: Deck) -> Optional[Bits]:
        # decode metadata
        step_size, start_padding, end_padding, lengths = self._decode_metadata(msg_metadata, len(msg))

        if step_size == 0:
            # no linear probing
            cards = msg
        else:
            # linear probing
            cards = self._unhash_msg(msg, step_size)
            agent_assert(len(cards) == len(lengths), NullDeckException(
                "ChunkConverter failed to decode deck to bits: " +
                f"expect message cards of length {len(lengths)}, but got {len(cards)} msg cards."))

        # decode message
        chunk_sizes = [
            self.max_chunk_size if length == '0' else self.max_chunk_size - 1
            for length in lengths
        ]
        bit_str = ''.join([
            '{0:b}'.format(card).zfill(chunk_sizes[i])
            for i, card in enumerate(cards)
        ])

        if end_padding > 0:
            bit_str = bit_str[start_padding:-end_padding] 
        else:
            bit_str = bit_str[start_padding:]

        return Bits(bin=bit_str)

    @classmethod
    def __str__(cls) -> str:
        return "ChunkConverter"

class PermutationConverter(BDC):
    """Converts between bits and deck of cards by mapping each permutation of
    cards to a unique integer in its binary representation."""

    def __init__(self, cards):
        super().__init__(cards)
        self.permuter = PermutationGenerator()

    def to_deck(self, bits: Bits) -> Optional[tuple[Deck, Deck]]:
        bit_len = len(bits.bin)

        num_msg_cards = self.permuter.n_needed(2 ** bit_len)
        num_metdata_cards = 6 # 6! = 720, can handle bit length up to 720

        if num_msg_cards + num_metdata_cards > len(self.cards):
            # PermutationConverter needs more cards than given to
            # encode the given bit pattern
            return None

        # TODO: use checksum instead of metadata for representing length of bits
        msg_cards = list(map(str, self.cards[-num_msg_cards:]))
        metadata_cards = list(map(str, self.cards[:num_metdata_cards]))

        msg = [
            int(card)
            for card in self.permuter.encode(msg_cards, bits.uint)
        ]
        msg_metadata = [
            int(card)
            for card in self.permuter.encode(metadata_cards, bit_len)
        ]

        return msg_metadata, msg

    def to_bits(self, msg: Deck, msg_metadata: Deck) -> Optional[Bits]:
        # TODO: dynamically detect message length
        agent_assert(len(msg_metadata) >= 6, NullDeckException(
                f"PermutationConverter expects message metadata of 6 cards, " +
                f"but got a message metadata deck of size {len(msg_metadata)}."))

        metadata_cards = list(map(str, msg_metadata[-6:]))
        cards = list(map(str, msg))

        metadata = self.permuter.decode(metadata_cards) # i.e. bit length
        bits = Bits(uint=self.permuter.decode(cards), length=metadata)
        debug(f"recovered message deck: {msg} -> {bits.bin}")

        return bits

    @classmethod
    def __str__(cls) -> str:
        return "PermutationConverter"

def to_partial_deck(
    partial_match: bool,
    domain:        Domain,
    msg_bits:      Bits,
    free_cards:    Deck
) -> Optional[tuple[BDC, Deck, Deck, Deck]]:
    """Dynamically select the best BDC to encode bits to deck.
    
    Returns None if all BDCs failed. Otherwise, returns the BDC that
    encodes the bits with the fewest number of cards, i.e. a tuple of
        (BDC, metadata deck, message metadata deck, message deck)
    """
    meta_codec = MetaCodec()

    def _to_partial_deck(bdc: BDC) -> Optional[tuple[Deck, Deck]]:
        metadata_deck = meta_codec.encode(partial_match, domain, bdc)

        # compute cards availabe for encoding the message, to avoid
        # collision between metadata cards and message cards
        free_msg_cards = [card for card in free_cards if card not in metadata_deck]
        msg_deck = bdc(free_msg_cards).to_deck(msg_bits)

        if msg_deck is None:
            info(f"{bdc.__str__()} can't encode bits",
                f"(msg_bits: {len(msg_bits.bin)}) to deck:", msg_bits.bin)

        return (bdc, metadata_deck, *msg_deck) if msg_deck is not None else None

    def _msg_deck_size(partial_deck: Optional[tuple[Deck, Deck]]) -> int:
        if partial_deck is None:
            return sys.maxsize
        
        bdc, metadata, msg_metadata, msg = partial_deck
        deck_sizes = [len(metadata), len(msg_metadata), len(msg)]
        info(f"{bdc.__str__()} encoded deck size: ",
            f"{sum(deck_sizes)} = {' + '.join(map(str, deck_sizes))}")

        return sum(deck_sizes)

    BDCs = [ChunkConverter, PermutationConverter]
    partial_decks = list(map(_to_partial_deck, BDCs))

    # if all BDCs failed to encode the bits to deck, return None
    for ans in partial_decks:
        if ans is not None:
            break
    else:
        return None

    # some BDC succeeded, select the one that uses the least number of cards
    sizes = list(map(_msg_deck_size, partial_decks))
    min_partial_deck = partial_decks[sizes.index(min(sizes))]

    return min_partial_deck


# -----------------------------------------------------------------------------
#   Bits <-> String Codec
# -----------------------------------------------------------------------------

class MetaCodec:
    """Codec for (domain, BDC) <-> deck"""
    def __init__(self):
        self.domains = [
                None, # padding
                Domain.PASSWORD, Domain.COORDS, # format specific
                Domain.ADDRESS, Domain.FLIGHTS, # format specific with dictionary
                Domain.WAR_WORDS, Domain.PLACES_AND_NAMES, Domain.SIX_WORDS, # dictionary domains
                None, Domain.ALPHA_NUMERIC, # format generic
                Domain.GENERIC #generic all
            ]
        self.BDCs = [ChunkConverter, PermutationConverter]
        self.domain_mappings = {
            Domain.PASSWORD: 1,
            Domain.COORDS: 2,
            Domain.ADDRESS: 3,
            Domain.FLIGHTS: 4,
            Domain.WAR_WORDS: 5,
            Domain.PLACES_AND_NAMES: 6,
            Domain.SIX_WORDS: 7,
            Domain.ALPHA_NUMERIC: 9,
            Domain.GENERIC: 10
        }


    def encode(self, partial_match: bool, domain: Domain, bdc: BDC) -> Deck:
        # use the index to represent domain and BDC
        domain_idx = self.domain_mappings[domain]
        bdc_idx = self.BDCs.index(bdc)

        # metadata bit representation:
        # domain (4 bits) + partial match (1 bit) + BDC (1 bit) => metadata (6 bits)
        domain_bits = Bits(uint=domain_idx, length=4)
        bdc_bit = Bits(uint=bdc_idx, length=1)
        partial_match_bit = Bits(uint=int(partial_match), length=1)
        metadata = Bits(bin=f'0b{domain_bits.bin}{partial_match_bit.bin}{bdc_bit.bin}')

        # Bit -> Card
        #   smallest of the least significant 5 bits is: _001 0 0 = 4
        #   largest  of the least significant 5 bits is: _111 1 1 = 31
        # if we split into metadata into 2 cards using 1 bit and 5 bits,
        # there won't be collision: 0-2 & 4-31
        deck = [int(metadata.bin[0], 2), int(metadata.bin[1:], 2)]

        # logging
        info('MetaCodec encode: %s (domain), %s (BDC) -> %s (deck)' % 
            (domain.name, bdc.__str__(), str(deck)))
        info("%5s %-5s %-8s %-5s %-7s %-5s %10s" %
            ("BDC", "BITS", "PARTIAL", "BITS", "DOMAIN", "BITS", "METADATA"))
        info( "%5d %-5s %-8d %-5s %-7d %-5s %10s" %
            (bdc_idx, bdc_bit.bin, int(partial_match), partial_match_bit.bin, domain_idx, domain_bits.bin, metadata.bin))

        return deck

    def decode(self, deck: Deck) -> tuple[bool, Domain, BDC]:
        """Decodes the metadata deck into a tuple of (partial_match, domain, bdc).
        
        Raises NullDeckException if decoding fails.
        """
        try:
            metadata = Bits(uint=deck[0] * 32 + deck[1], length=6)
            bdc_bit, partial_match_bit, domain_bits = metadata.bin[-1], metadata.bin[-2], metadata.bin[:-2]

            bdc_idx = int(bdc_bit, 2)
            partial_match = bool(int(partial_match_bit, 2))
            domain_idx = int(domain_bits, 2)

            bdc = self.BDCs[bdc_idx]
            domain = self.domains[domain_idx]
        except Exception as e:
            raise NullDeckException(f"can't decode metadata: {e}")
        
        if domain is None:
            raise NullDeckException("can't deccode metadata: domain in metadata is non-existent")

        info('MetaCodec decode:  %s (deck) -> %s (domain), %s (BDC)' %
            (str(deck), domain.name, bdc.__str__()))
        info("%5s %-5s %-8s %-5s %-7s %-5s %10s" %
            ("BDC", "BITS", "PARTIAL", "BITS", "DOMAIN", "BITS", "METADATA"))
        info( "%5d %-5s %-8d %-5s %-7d %-5s %10s" %
            (bdc_idx, bdc_bit, int(partial_match), int(partial_match), domain_idx, domain_bits, metadata.bin))

        return partial_match, domain, bdc


class Huffman:

    def __init__(self, dictionary: Optional[List[str]] = None) -> None:
        if dictionary is not None:
            self.codec = HuffmanCodec.from_frequencies(dictionary)
        else:
            self.codec = load_shakespeare()
        toTruncate = all([bin(val)[2:].rjust(bits, '0')[0]=="0" for symbol, (bits, val) in self.codec.get_code_table().items() if val !=  1])
        self.modified_codec = {symbol: (bin(val)[2:].rjust(bits-1, '0') if toTruncate else bin(val)[2:].rjust(bits, '0')) for symbol, (bits, val) in self.codec.get_code_table().items()}
 
    def _add_padding(self, msg: Bits, padding_len: int) -> Bits:

        padding_bits = '{0:b}'.format(0).zfill(
            padding_len) if padding_len > 0 else ''
        padded_msg_bin = '0b{}{}'.format(padding_bits, msg.bin)

        padded_msg = Bits(bin=padded_msg_bin)

        debug('[ Huffman._add_padding ]',
              f'len(msg): {len(msg)}, msg: {msg.bin}',
              f'padding size: {padding_len}', f'padded msg: {padded_msg.bin}')

        return padded_msg

    def _remove_padding(self, msg: Bits, padding_len: int) -> Bits:
        original_encoding = Bits(bin=f'0b{msg.bin[padding_len:]}')

        debug('[ Huffman._remove_padding ]',
              f'original encoding: {original_encoding.bin}')

        return original_encoding

    def encode(self, msg: str, padding_len: int = 5) -> Bits:
        encoded_message = []
        for char in msg:
            if not char in self.modified_codec:
                raise ValueError()
            encoded_message.append(self.modified_codec[char])
        bits = Bits(bin=''.join(encoded_message))

        # What we use to encode to
        # old_bits = Bits(bytes=self.codec.encode(msg))
        # debug(f'old_bits: {len(old_bits.bin)}, new_bits: {len(bits.bin)}')
        debug('[ Huffman.encode ]', f'msg: {msg} -> bits: {bits.bin}')

        padded_bits = self._add_padding(bits, padding_len)
        return padded_bits

    def decode(self, bits: Bits, padding_len: int = 5) -> str:
        decoded_message = ""
        bits = self._remove_padding(bits, padding_len)
        encoded_message = bits.bin

        while encoded_message:
            for symbol, code in self.modified_codec.items():
                if encoded_message.startswith(code):
                    decoded_message += symbol
                    encoded_message = encoded_message[len(code):]
                    break
            else:
                # since now for partial match, we truncate a bit pattern
                # to a fixed length. After the truncation, the bit pattern
                # may no longer be a valid huffman decoding.
                error(f"huffman can't decode {encoded_message}")
                break
            #     raise ValueError()
        # What we used to decode to (doesn't work bc new encoding)
        # old_decoded = self.codec.decode(bits.tobytes())

        debug('[ Huffman.decode ]', f'bits: {bits.bin} -> msg: {decoded_message}')

        return decoded_message


# -----------------------------------------------------------------------------
#   Group 3 Agent
# -----------------------------------------------------------------------------

class Agent:

    def __init__(self) -> None:
        self.stop_card = 51

        # NOTE: this is coupled with MetadataCodec.encode
        self.free_cards = list(range(0, 32))
        self.trash_cards = list(range(32, 51))

        # IMPORTANT: ordering here matters since we check them linearlly
        # a subset of a less efficient one should be found sooner (ie alpha numeric last since WAR_WORDS is a more efficient subset)
        self.domain_detector = DomainDetector(
            [
                Domain.PASSWORD, Domain.COORDS, # format specific
                Domain.ADDRESS, Domain.FLIGHTS, # format specific with dictionary
                Domain.SIX_WORDS, Domain.WAR_WORDS, Domain.PLACES_AND_NAMES, # dictionary domains
                Domain.ALPHA_NUMERIC # format generic
            ]
        )

        self.domain2transformer = {
            Domain.GENERIC: WordTransformer(),
            Domain.PASSWORD: PasswordsTransformer(),
            Domain.COORDS: CoordsTransformer(),
            Domain.ADDRESS: AddressTransformer(),
            Domain.FLIGHTS: FlightsTransformer(),
            Domain.WAR_WORDS: WarWordsTransformer(),
            Domain.PLACES_AND_NAMES: PlacesAndNamesTransformer(),
            Domain.SIX_WORDS: SixWordsTransformer(),
            Domain.ALPHA_NUMERIC: AlphaNumericTransformer()
        }

        # TODO
        # bit legnths to partial match
        # Permutation Converter:
        #   32 - 2 (metadata) - 6 (message metadata) = 24 cards to use for message
        #   argmax(n) of n! - 2^24 => 79
        #                     2^25 => 83
        #                     2^26 => 88
        #                     2^27 => 93
        #                     2^28 => 97
        self.partial_match_len = [79]
 
    # TODO: there might be many _tangle_cards and _untangle_cards methods
    # such as using checksum vs using trashcards, so this should be extracted
    # as a component that could be swapped and reused.
    def _tangle_cards(self, metadata: Deck, message_metadata: Deck, message: Deck) -> Deck:
        used_cards = self.trash_cards + [self.stop_card] + metadata + message + message_metadata
        unused_message_cards = [
            card for card in range(0, 52)
            if card not in used_cards
        ]

        deck = (self.trash_cards
              + unused_message_cards
              + message_metadata
              + [self.stop_card]
              + message
              + metadata)

        return deck if valid_deck(deck) else list(range(52))

    def _untangle_cards(self, cards: Deck) -> tuple[Deck, Deck, Deck]:
        """Untangles deck of 52 cards into metadata deck, msg metadata deck, and msg deck
        after removing trash cards.
        
        Raises NullDeckException if the decks untangled are invalid, e.g. deck size incorrect.
        """
        def remove_trash_cards(deck) -> List[int]:
            deck = deck.copy()
            for i in self.trash_cards:
                deck.remove(i)
            return deck

        deck = remove_trash_cards(cards)
        stop_card = deck.index(self.stop_card)
        message_metadata, message, metadata = deck[:stop_card], deck[stop_card+1:-2], deck[-2:]

        # estimate the number of shuffles through the position of the deck in the received deck
        # if it's at the very top, then we are screwed, there's no hope of recovering the message,
        # so can safely raise a NullDeckException
        stop_card_orig_pos, msg_metadata_len = cards.index(self.stop_card), len(message_metadata)
        agent_assert(stop_card_orig_pos >= msg_metadata_len, NullDeckException(
            "received deck either doesn't contain a message or gets shuffled too many times: " +
            f"stop card at index {stop_card_orig_pos}, but have msg metadata of {msg_metadata_len} cards"
        ))

        agent_assert(len(message) > 0, NullDeckException("empty message"))
        agent_assert(len(metadata) == 2,
            NullDeckException(f"expect 2 metadata cards, but got {len(metadata)}"))

        return metadata, message_metadata, message

    def _encode_bits_to_deck(
        self,
        bits:       Bits,
        domain:     Domain,
        partial:    bool,
        free_cards: Deck
    ) -> tuple[Bits, Optional[List[int]]]:

        partial_deck = to_partial_deck(partial, domain, bits, free_cards)
        if partial_deck is None:
            info(f"bits too long, can't encode to deck: {bits.bin}")
        
        return partial_deck

    def encode(self, msg: str) -> List[int]:
        # Encoding Steps
        # 1. detect the domain of the message, if none matches, fall back to
        #    using Domain.GENERIC
        # 2. based on the domain, choose a domain-specific *MessageTransformer*
        #    to compress the message string into a shorter one
        # 3. dynamically select a *Bits <-> Deck Converter* based on the
        #    shortened message, returns two decks:
        #      a) metadata deck: containing domain and BDC type
        #      b) message deck: contains the encoded message
        # 4. tangles the decks, trash cards, unused cards, stop cards into a
        #    final deck returned to the simulator

        info(f"\n[ {msg} ] encode")

        # step 1: domain detection
        domain = self.domain_detector.detect(msg)
        info(f"[ {msg} ]", f"domain: {domain.name}")

        # step 2: message str -> compressed msg str -> bits
        compressed_msg, bits = self.domain2transformer[domain].compress(msg)
        info(f"[ {msg} ]",
            f"transformer: {self.domain2transformer[domain]},",
            f"compressed message: \"{compressed_msg}\",",
            f"bits: {bits.bin}")

        # step 3 (self._encode): bits -> partial deck
        partial_match = False
        free_cards = self.free_cards
        partial_deck = self._encode_bits_to_deck(bits, domain, partial_match, free_cards)

        if partial_deck is None:
            info(f"[ {msg} ]", "trying partial match by truncating the bit patterns...")

        i = 0
        # try for partial match by truncating the bits of the original message
        # TODO: should we truncate bit patterns or truncate the original message?
        partial_match = True
        while partial_deck is None and i < len(msg):
            bit_len = self.partial_match_len[i]
            bits = Bits(bin=bits.bin[:bit_len])

            partial_deck = self._encode_bits_to_deck(bits, domain, partial_match, free_cards)
            i += 1

        if partial_deck is None:
            info(f"[ {msg} ]", "msg too long, can't encode to deck")
            return list(range(51,-1,-1))

        # step 4: bits -> partial deck success => produce final deck
        bdc, metadata_cards, message_metadata_cards, message_cards = partial_deck
        info(f"[ {msg} ]", f"cards available for encoding message: {free_cards}")
        info(f"[ {msg} ] bits <-> deck converter: {bdc.__str__()}:",
            f"partial match: {partial_match}, ",
            f"metadata cards: {metadata_cards},",
            f"message metadata cards: {message_metadata_cards},",
            f"message cards: {message_cards},")

        final_deck = self._tangle_cards(metadata_cards, message_metadata_cards, message_cards)
        info(f"[ {msg} ]", f"valid deck: {valid_deck(final_deck)}, ", f"final deck: {final_deck}")

        return final_deck

    def decode(self, deck: Deck) -> str:
        # Decoding Steps:
        # 1. recover the deck that contains the message and metadata
        #    The metadata containing the bdc and domain is fixed size and are always at the end of the deck of message + metadata
        #    We can Use a single card to represent the metdata which is 4 bits
        #        3 bits for MessageTransformer + 1 bits for bdc
        # 2. read the metadata to recover bdc and domain
        # 3. use bdc to convert message deck -> message bits
        # 4. use domain to convert message bits -> original message string
        info(f"\ndecode, deck: {deck}")

        # if an error occurs during decoding, it means the message is corrupted
        # or the deck doesn't contain a message, we catch the error and return
        # the NULL_MESSAGE.
        try:
            metadata, message_metadata, message = self._untangle_cards(deck)
            info(f"untangled deck: ",
                f"metadata cards: {metadata},",
                f"message metadata cards: {message_metadata},",
                f"message cards: {message}")

            partial_match, domain, bdc = MetaCodec().decode(metadata)
            info(f"decoded metadata: partial match: {partial_match},",
                f"domain: {domain.name}, bits <-> deck converter: {bdc.__str__()}")

            # deck -> message bits
            free_msg_cards = [card for card in self.free_cards if card not in metadata]
            message_bits = bdc(free_msg_cards).to_bits(message, message_metadata)
            info(f"deck -> bits using {bdc.__str__()}: {message_metadata} -> {message_bits.bin}")

            # TODO: domain specific agent_assert's
            orig_msg = self.domain2transformer[domain].uncompress(message_bits)
            if partial_match:
                orig_msg = "PARTIAL: " + orig_msg
            info(f"using transformer: {self.domain2transformer[domain]},",
                f"uncompressed message: \"{orig_msg}\"")
        except NullDeckException as e:
            error("(NullDeckException)", e)
            orig_msg = NULL_MESSAGE
        except Exception as e:
            error(f"({type(e).__name__})", f"decoding failed with uncaught exeception: {e}")
            orig_msg = NULL_MESSAGE
        finally:
            return orig_msg


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

def test_metacodec():
    codec = MetaCodec()

    for domain, bdc in [
        (Domain.GENERIC, ChunkConverter),
        (Domain.PASSWORD, PermutationConverter)
    ]:
        deck = codec.encode(domain, bdc)
        codec.decode(deck)


if __name__ == "__main__":
    test_metacodec()
    test_huffman_codec()