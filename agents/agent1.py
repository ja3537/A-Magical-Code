import hashlib
import heapq
import logging
import math
from collections import deque
from copy import deepcopy

import numpy as np

import cards


log_format = "%(levelname)s: %(funcName)s(): %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)
# logger.disabled = True

# Permutations of input strings
alpha = " abcdefghijklmnopqrstuvwxyz"
numeric = " 0123456789"
alpha_numeric = alpha + numeric[1:]  # Don't include space twice
alpha_numeric_punc = alpha_numeric + "."


def calc_checksum(num: int, mod_prime=40213, mode="blake2b", base=239):
    """Calculate the checksum from an interger repr the binary data
    Args:
        num: Decimal representing message
        mod_prime: Maximum value of the checksum. Better if it's a prime number.
            Values: [113, 719, 4973, 40213, 362867]
        mode: Which checksum to use (blake or polynomial)
        base: The base of the number system. Should be equal to the len of the message sequence.
    """
    if mode not in ["blake2b", "polynomial"]:
        raise ValueError

    # Convert bit string into a series of bytes
    length = (max(num.bit_length(), 1) + 7) // 8
    byte_seq = num.to_bytes(length, "big")  # Max possible message = 12 bytes long (after compression)

    if mode == "polynomial":
        checksum = 0
        for n in byte_seq:
            checksum = ((checksum + n) * base) % mod_prime
    else:
        h = hashlib.blake2b(digest_size=2)
        h.update(byte_seq)  # Max value of card = 51 < 1 byte
        hash = int.from_bytes(h.digest(), "little")
        checksum = hash % mod_prime
    return checksum


# ------- str to perm and vice-versa ----------------- #
class Perm:
    def __init__(self, valid_cards=tuple(range(52 - 12, 52)), valid_char_str=alpha):
        """Borrowed and modified from group 7"""
        self.encoding_len = len(valid_cards)
        self.max_msg_len = math.floor(math.log(math.factorial(self.encoding_len), len(valid_char_str)))
        self.perm_zero = valid_cards
        factorials = [0] * self.encoding_len
        for i in range(self.encoding_len):
            factorials[i] = math.factorial(self.encoding_len - i - 1)
        self.factorials = factorials
        self.char_list = valid_char_str

    def check_num_too_large(self, num):
        items = list(self.perm_zero[:])
        f = self.factorials[0]
        lehmer = num // f
        if lehmer > len(items)-1:
            return True
        else:
            return False

    def num_to_perm(self, n):
        if self.check_num_too_large(n):
            logger.warning(f"Input text too long to encode into {self.encoding_len} cards.")
            return []

        perm = []
        items = list(self.perm_zero[:])
        for idx, f in enumerate(self.factorials):
            lehmer = n // f
            perm.append(items.pop(lehmer))
            n %= f
        return perm

    def str_to_num(self, message):
        """Convert a message string into a decimal"""
        # Stop match string at unknown char, to meet with partial requirements
        tokens = []
        for ch in message:
            if ch in self.char_list:
                tokens.append(ch)
            else:
                break
        # while len(tokens) < self.max_msg_len:
        #     tokens.append(" ")
        tokens = tokens[::-1]

        if len(message) > len(tokens):
            logger.warning(f"Invalid characters found in message '{message}. "
                           f"Clipping message to '{''.join(tokens[::-1])}'")

        # Convert char to int
        num = 0
        final_tokens = []
        for idx, ch in enumerate(tokens):
            num_ = self.char_list.index(ch) * len(self.char_list) ** idx
            # if (num + num_) // self.factorials[0] > len(self.perm_zero):
            #     logger.warning(f"Input text too long to encode into {self.encoding_len} cards. "
            #                    f"Shortening message to '{''.join(final_tokens[::-1])}'")
            #     break
            num += num_
            final_tokens.append(ch)

        return num

    def str_to_perm(self, message):
        """Convert a message string into a sequence of cards"""
        # TODO: Add notation for unknown chars. Make it a partial match
        # TODO: If there is a space exactly where the msg is cut off, we don't recognize that as a partial match
        max_chars = self.max_msg_len
        if len(message) > max_chars:
            message = message[:max_chars]
            logger.warning(f"Input text longer than {max_chars} characters. Shortening message to "
                           f"'{message}'")

        num = self.str_to_num(message)
        perm = self.num_to_perm(num)
        return perm

    def perm_to_num(self, permutation):
        """Convert a sequence of cards into a decimal number"""
        n = len(permutation)
        number = 0

        for i in range(n):
            k = 0
            for j in range(i + 1, n):
                if permutation[j] < permutation[i]:
                    k += 1
            number += k * self.factorials[i]
        return number

    def num_to_str(self, num):
        """Convert a decimal number into a string of characters.
        Treat the characters as a number system of base (num of unique characters)"""
        words = []
        break_next = False
        while True:
            num, index = divmod(num, len(self.char_list))
            words.append(self.char_list[index])
            if break_next:
                break
            if num == 0:
                break_next = True
        return ''.join(words[::-1]).strip()

    def perm_to_str(self, perm):
        """Convert a sequence of cards into a message string"""
        num = self.perm_to_num(perm)
        msg = self.num_to_str(num)
        return msg


# --------------------------- huffman_decoding --------------------------- #
class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        # frequency of symbol
        self.freq = freq
        # symbol name (character)
        self.symbol = symbol
        # node left of current node
        self.left = left
        # node right of current node
        self.right = right

        # tree direction (0/1)
        self.huff = ''
        self.encoding_dict = {}
        self.build_encoding_dict()

    def __lt__(self, nxt):
        return self.freq < nxt.freq

    def build_encoding_dict(self, val='', node=None):
        """Utility function to print huffman codes for all symbols in the newly created Huffman tree"""
        # huffman code for current node
        if node is None:
            node = self
        newVal = val + str(node.huff)

        # if node is not an edge node then traverse inside it
        if node.left:
            self.build_encoding_dict(newVal, node.left)
        if node.right:
            self.build_encoding_dict(newVal, node.right)

        # if node is edge node then display its huffman code
        if not node.left and not node.right:
            self.encoding_dict[node.symbol] = newVal

    def print_codes(self):
        for key, val in self.encoding_dict.items():
            print(key, " -> ", val)


class Huffman:
    def __init__(self):
        # characters for huffman tree
        self.chars = ['e', 'm', 'a', 'h', 'r', 'g', 'i', 'b', 'o', 'f', 't', 'y', 'n', 'w', 's',
                      'k', 'l', 'v', 'c', 'x', 'u', 'z', 'd', 'j', 'p', 'q',
                      '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', ]

        # frequency of characters
        # Ref: https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
        self.freq = [11.1607, 3.0129, 8.4966, 3.0034, 7.5809, 2.4705, 7.5448, 2.072, 7.1635,
                     1.8121, 6.9509, 1.7779, 6.6544, 1.2899, 5.7351, 1.1016, 5.4893, 1.0074,
                     4.5388, 0.2902, 3.6308, 0.2722, 3.3844, 0.1965, 3.1671, 0.1962,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.11, ]

        # list containing unused nodes
        nodes = []

        # converting characters and frequencies
        # into huffman tree nodes
        for x in range(len(self.chars)):
            heapq.heappush(nodes, Node(self.freq[x], self.chars[x]))

        while len(nodes) > 1:
            # sort all the nodes in ascending order
            # based on their frequency
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)

            # assign directional value to these nodes
            left.huff = 0
            right.huff = 1

            # combine the 2 smallest nodes to create
            # new node as their parent
            new_node = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)

            heapq.heappush(nodes, new_node)

        # Huffman Tree is ready!
        self.nodes = nodes
        self.encoding_dict = nodes[0].encoding_dict

        self.decoding_dict = {}
        for letter, binstr in self.encoding_dict.items():
            self.decoding_dict[binstr] = letter

    def print_codes(self):
        self.nodes[0].print_codes()

    def encode(self, message):
        encoding = ""
        clipped = False
        for letter in message:
            try:
                encoding += self.encoding_dict[letter]
            except KeyError:
                logger.debug(f"Unsupported char '{letter}' in message '{message}'. Clipping.")
                clipped = True
                break
        return encoding, clipped

    def decode(self, encoded_data):
        huffman_tree = self.nodes[0]
        tree_head = huffman_tree
        decoded_output = []
        for x in encoded_data:
            if x == '1':
                huffman_tree = huffman_tree.right
            elif x == '0':
                huffman_tree = huffman_tree.left
            else:
                raise ValueError(f"Encoding must be binary. Got: {encoded_data}")

            if huffman_tree.left is None and huffman_tree.right is None:
                decoded_output.append(huffman_tree.symbol)
                huffman_tree = tree_head
        string = ''.join([str(item) for item in decoded_output])
        return string

    @staticmethod
    def encoding_to_num(encoding: str):
        for ch in encoding:
            if not (ch == "0" or ch == "1"):
                raise ValueError(f"Encoding must be binary. Got: {encoding}")

        return int(encoding, 2)

    @staticmethod
    def num_to_binstr(num: int) -> str:
        # helper func to convert an int to a binary str
        return bin(num)[2:]


class Agent:
    def __init__(self):
        self.seed = 0
        # self.rng = np.random.default_rng(self.seed)

        # Checksum
        self.checksum_bits = 16
        self.mod_prime = 65423  # 16 bits
        self.mod_mode = "blake2b"

        # Message
        self.encode_len = 22  # Max num of cards in seq
        self.char_set = alpha
        self.huff = Huffman()
        # self.max_msg_len = math.floor(math.log(math.factorial(self.encode_len), len(self.char_set)))
        self.max_msg_bits = math.floor(math.log(math.factorial(self.encode_len), 2))
        self.max_msg_bits -= self.checksum_bits + 2  # partial bit and pad bit
        self.valid_cards_p = tuple(range(52 - self.encode_len, 52))
        self.perm = Perm(self.valid_cards_p, self.char_set)

        # Unscrambling
        self.dque = deque([])
        self.max_trials = len(self.char_set) ^ 3 + 1

    def pad_bitstr(self, bitstr: str, length: int):
        """Pad a bit str with leading zeros to req length"""
        lb = len(bitstr)
        if lb > length:
            raise ValueError(f"Bitstr longer than length")
        bit = "0"*(length - lb) + bitstr
        return bit

    def encode(self, message_):
        message = deepcopy(message_)
        partial = "0"

        while True:
            bit_msg, clipped = self.huff.encode(message)
            if len(bit_msg) <= self.max_msg_bits:
                break
            else:
                message = message[:-1]

        if len(message) < len(message_):
            partial = "1"
            logger.debug(f"Input text longer than {self.max_msg_bits} bits. Shortening message to '{message}'")

        msg_num = int(bit_msg, 2)
        checksum = calc_checksum(msg_num, mod_prime=self.mod_prime, mode=self.mod_mode)
        bit_checksum = self.pad_bitstr(bin(checksum)[2:], self.checksum_bits)
        bit_total = partial + bit_checksum + bit_msg
        bit_total = "1" + bit_total  # avoid confusion with leading zeros
        num_total = int(bit_total, 2)
        seq_encode = self.perm.num_to_perm(num_total)

        # Scramble the rest of the cards above message
        seq_rand = [c for c in range(52) if c not in self.valid_cards_p]
        # self.rng.shuffle(seq_rand)
        seq_total = seq_rand + seq_encode

        if not cards.valid_deck(seq_total):
            raise ValueError(f"Not a valid deck from encode?")

        return seq_total

    def verify_msg(self, deck):
        num = self.perm.perm_to_num(deck)
        bit_total = bin(num)[2:]
        if bit_total[1] == "1":
            partial = True
        else:
            partial = False
        bit_total = bit_total[2:]  # Ignore the extra bits we added to avoid confusion with leading zeros

        bit_checksum = bit_total[:self.checksum_bits]
        bit_msg = bit_total[self.checksum_bits:]

        checksum_decoded = int(bit_checksum, 2)
        num_msg = int(bit_msg, 2)
        checksum_calculated = calc_checksum(num_msg, mod_prime=self.mod_prime, mode=self.mod_mode)
        if checksum_calculated == checksum_decoded:
            msg = self.huff.decode(bit_msg)
            if partial:
                msg = "PARTIAL: " + msg
            return msg
        else:
            return None

    @staticmethod
    def deshuffle1(deck):
        """Gets a list of decks by moving each card to the top"""
        ds_decks = []
        for idx in range(1, len(deck)):
            deck_in = deck.copy()
            top_card = deck_in.pop(idx)
            deck_in.insert(0, top_card)
            ds_decks.append(deck_in)
        return ds_decks

    def decode(self, deck):
        seq_encode = [c for c in deck if c in self.valid_cards_p]

        # Try to recover message (unscramble if necessary)
        max_trials = len(self.char_set) ^ 3 + 1  # Greater than depth 3 is ineffective (from our tests)
        dque = deque([])
        dque.append(seq_encode)
        decoded_str = "NULL"
        while max_trials > 0 and len(dque) > 0:
            ddeck = dque.popleft()
            msg = self.verify_msg(ddeck)
            if msg is not None:
                decoded_str = msg
                break
            else:
                dque.extend(self.deshuffle1(ddeck))
                max_trials -= 1

        # TODO: Add case for partial strings
        return decoded_str


if __name__ == "__main__":
    logger.disabled = False
    logging.basicConfig(level=logging.DEBUG, format=log_format)
    test_agent = False
    test_perm = False
    test_huff = False

    # Testing Agent
    # test_agent = True
    if test_agent:
        agent = Agent()
        msg = "abc"
        deck = agent.encode(msg)
        if not cards.valid_deck(deck):
            raise ValueError

        num_shuffles = 10
        rng = np.random.default_rng()
        shuffles = rng.integers(0, 52, num_shuffles)
        for pos in shuffles:
            top_card = deck[0]
            deck = deck[1:]
            deck = deck[:pos] + [top_card] + deck[pos:]

        msg_decoded = agent.decode(deck)
        print(f"Message: {msg}")
        print(f"Shuffled deck: {deck}")
        print(f"Decoded Message: {msg_decoded}")

    # Testing the str-to-perm and back
    # test_perm = True
    if test_perm:
        encode_len = 12
        valid_cards = tuple(range(52 - encode_len, 52))
        test_perm = Perm(valid_cards, alpha)
        card_perm = test_perm.str_to_perm("hel loworld")
        print("card seq = " + str(card_perm))
        decoded_str = test_perm.perm_to_str(card_perm)
        print("recovered string = " + decoded_str)

    # test_huff = True
    if test_huff:
        huff = Huffman()
        huff.print_codes()

        msg = "byouth"
        encoding, clipped = huff.encode(msg)
        print(f"Encoding - {msg}: {encoding}")
        decoded_msg = huff.decode(encoding)
        print(f"Decoding - {encoding}: {decoded_msg}")
        print(f"Decoding Test: {decoded_msg == msg}")

    print("Done")

