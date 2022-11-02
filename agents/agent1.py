import heapq
import logging
import math
import random
from itertools import permutations

import numpy as np
from dahuffman import HuffmanCodec

import cards


logging.disable(logging.INFO)


def calc_checksum(number, base=127):
    """Calculate the checksum from an interger repr the binary data
    Args:
        number: Int corresponding to binary data of the input message
        base: Should be equal to the size of character set used in the message
    """
    num_bin = bin(number)[2:]
    chunk_len = 5
    checksum = 0
    mod_prime = 113
    # From wikipedia - rollin hash
    # ASCII a = 97, b = 98, r = 114.
    # hash("abr") =  [ ( [ ( [  (97 × 256) % 101 + 98 ] % 101 ) × 256 ] %  101 ) + 114 ]   % 101   =  4
    while len(num_bin) > 0:
        bin_chunk = num_bin[:chunk_len]
        num_bin = num_bin[chunk_len:]

        num_chunk = int(bin_chunk, 2)
        checksum = (checksum + num_chunk) % mod_prime
        if len(num_bin) > 0:
            checksum = (checksum * base) % mod_prime
    return checksum

# ------- str to perm and vice-versa ----------------- #
"""Borrowed and modified from group 7"""

MAX_CHAR = 10  # Max num of char in input message

alpha = " abcdefghijklmnopqrstuvwxyz"
numeric = " 0123456789"
alpha_numeric = alpha + numeric[1:]  # Don't include space twice
alpha_numeric_punc = alpha_numeric + "."


class Perm:
    def __init__(self, valid_cards=tuple(range(52 - 12, 52)), valid_char_str=alpha):
        self.encoding_len = len(valid_cards)
        self.perm_zero = valid_cards
        factorials = [0] * self.encoding_len
        for i in range(self.encoding_len):
            factorials[i] = math.factorial(self.encoding_len - i - 1)
        self.factorials = factorials
        self.char_list = valid_char_str

    def perm_to_num(self, permutation):
        n = len(permutation)
        # s = sorted(permutation)
        number = 0

        for i in range(n):
            k = 0
            for j in range(i + 1, n):
                if permutation[j] < permutation[i]:
                    k += 1
            number += k * self.factorials[i]
        return number

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
            logging.warning(f"Input text too long to encode into {self.encoding_len} cards.")
            return []

        perm = []
        items = list(self.perm_zero[:])
        for idx, f in enumerate(self.factorials):
            lehmer = n // f
            perm.append(items.pop(lehmer))
            n %= f
        return perm

    def str_to_num(self, message, max_chars=12):
        # Stop match string at unknown char, to meet with partial requirements
        tokens = []
        for ch in message:
            if ch in self.char_list:
                tokens.append(ch)
            else:
                break
        while len(tokens) < max_chars:
            tokens.append(" ")
        tokens = tokens[::-1]

        # Convert char to int
        max_trials = 100
        while max_trials > 0:
            max_trials -= 1
            num = 0
            for idx, ch in enumerate(tokens):
                num += self.char_list.index(ch) * len(self.char_list) ** idx

            # Check if message can fit in N cards
            if num // self.factorials[0] < len(self.perm_zero):
                break
            else:
                tokens = tokens[1:]
        if max_trials < 100 and tokens[0] != " ":
            logging.warning(f"Input text too long to encode into {self.encoding_len} cards. Shortening message to "
                            f"'{''.join(tokens[::-1])}'")
        return num

    def str_to_perm(self, message, max_chars=12):
        # TODO: Add notation for unknown chars. Make it a partial match
        # TODO: If there is a space exactly where the msg is cut off, we don't recognize that as a partial match
        if len(message) > max_chars:
            message = message[:max_chars]
            logging.warning(f"Input text longer than {max_chars} characters. Shortening message to "
                            f"'{message}'")

        num = self.str_to_num(message, max_chars)
        perm = self.num_to_perm(num)
        return perm

    def perm_to_str(self, perm):
        num = self.perm_to_num(perm)
        words = []
        break_next = False
        while True:
            index = num % len(self.char_list)
            words.append(self.char_list[index])
            num = num // len(self.char_list)
            if break_next:
                break
            if num == 0:
                break_next = True
        return ''.join(words[::-1]).strip()


class Unscramble:
    def __init__(self, card_deck, check_sum, perm_class: Perm, max_trials=10000) -> None:
        """Will unscramble the input deck until it matches the checksum
        Will terminate after max num of trials"""
        self.card_deck = card_deck
        self.check_sum = check_sum
        self.trials = max_trials
        self.perm = perm_class
        self.answer = False
        self.result = None

    def unscramble(self):
        self.recrusion([], self.card_deck)
        return self.result

    def recrusion(self, prev_deck: list[int], rest_deck: list[int]):
        if len(rest_deck) < 1:
            new_order = [rest_deck[0]] + prev_deck
            num = self.perm.perm_to_num(new_order)
            if calc_checksum(num, len(self.perm.char_list)) == self.check_sum:
                self.answer = True
                self.result = new_order
            return None
        if self.trials == 0 or self.answer:
            return None
        for i in range(len(rest_deck) - 1):
            if len(rest_deck) > 1:
                new_order = [rest_deck[i]] + prev_deck + rest_deck[0:i] + rest_deck[i + 1:]
                # In some combinations, the number might be too large
                # TODO: Check that this isn't an error
                try:
                    num = self.perm.perm_to_num(new_order)
                except IndexError:
                    num = 0

                if calc_checksum(num, len(self.perm.char_list)) == self.check_sum:
                    self.answer = True
                    self.result = new_order
                    break
        if not self.result:
            for i in range(len(rest_deck) - 1):
                new_prev = [rest_deck[i]] + prev_deck
                new_rest = rest_deck[0:i] + rest_deck[i + 1:]
                return self.recrusion(new_prev, new_rest)

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
                      'k', 'l', 'v', 'c', 'x', 'u', 'z', 'd', 'j', 'p', 'q', '/']

        # frequency of characters
        # Ref: https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
        self.freq = [11.1607, 3.0129, 8.4966, 3.0034, 7.5809, 2.4705, 7.5448, 2.072, 7.1635,
                     1.8121, 6.9509, 1.7779, 6.6544, 1.2899, 5.7351, 1.1016, 5.4893, 1.0074,
                     4.5388, 0.2902, 3.6308, 0.2722, 3.3844, 0.1965, 3.1671, 0.1962, 0.0001]

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

    def print_codes(self):
        self.nodes[0].print_codes()

    def encode(self, message):
        encoding = ""
        for letter in message:
            encoding += self.encoding_dict[letter]
        return encoding

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
        self.encode_len = 12  # Max num of cards in seq
        self.seed = 0
        self.rng = np.random.default_rng(self.seed)

        self.char_set = alpha
        self.max_msg_len = math.floor(math.log(math.factorial(self.encode_len), len(self.char_set)))
        self.valid_cards_p = tuple(range(52 - self.encode_len, 52))
        self.perm = Perm(self.valid_cards_p, self.char_set)

        self.cksum_len = 6
        self.valid_cards_c = tuple(range(52 - self.encode_len - self.cksum_len, 52 - self.encode_len))
        self.perm_ck = Perm(self.valid_cards_c, numeric)
        self.checksum_encode = None
        self.checksum_decode = None

    # def calc_checksum(self, number):
    #     """Number corresponding to the input message"""
    #     num_bin = bin(number)[2:]
    #     chunk_len = 5
    #     checksum = 0
    #     mod_prime = 113
    #     base = len(self.char_set)
    #     # ASCII a = 97, b = 98, r = 114.
    #     # hash("abr") =  [ ( [ ( [  (97 × 256) % 101 + 98 ] % 101 ) × 256 ] %  101 ) + 114 ]   % 101   =  4
    #     while len(num_bin) > 0:
    #         bin_chunk = num_bin[:chunk_len]
    #         num_bin = num_bin[chunk_len:]
    #
    #         num_chunk = int(bin_chunk, 2)
    #         checksum = (checksum + num_chunk) % mod_prime
    #         if len(num_bin) > 0:
    #             checksum = (checksum * base) % mod_prime
    #
    #     return checksum

    def encode(self, message):
        max_chars = self.max_msg_len
        if len(message) > max_chars:
            message = " " + message[:max_chars-1]
            logging.warning(f"Input text longer than {max_chars} characters. Shortening message to "
                            f"'{message}'")

        num = self.perm.str_to_num(message, max_chars)
        seq_encode = self.perm.num_to_perm(num)

        # Calculate Checksum
        checksum = calc_checksum(num, len(self.char_set))
        self.checksum_encode = checksum
        seq_checksum = self.perm_ck.num_to_perm(checksum)

        # Scramble the rest of the cards above message
        seq_rand = [c for c in range(52) if c not in self.valid_cards_p+self.valid_cards_c]
        self.rng.shuffle(seq_rand)
        seq_total = seq_rand + seq_encode + seq_checksum
        return seq_total

    def decode(self, deck):
        # Todo: Add checksum
        seq_encode = [c for c in deck if c in self.valid_cards_p]

        # TESTING
        seq_encode = seq_encode[1:] + [seq_encode[0]]

        seq_checksum = tuple([c for c in deck if c in self.valid_cards_c])
        decoded_checksum = self.perm_ck.perm_to_num(seq_checksum)
        self.checksum_decode = decoded_checksum

        msg_int = self.perm.perm_to_num(seq_encode)
        msg_checksum = calc_checksum(msg_int, len(self.perm.char_list)) + 1
        if msg_checksum == decoded_checksum:
            decoded_str = self.perm.perm_to_str(seq_encode)
        else:
            # Message has been scrambled. Try to recover
            unscramble = Unscramble(seq_encode, decoded_checksum, self.perm, max_trials=1000000)
            fixed_seq = unscramble.unscramble()

            if fixed_seq is not None:
                decoded_str = self.perm.perm_to_str(fixed_seq)
            else:
                decoded_str = "NULL"

        return decoded_str


if __name__ == "__main__":
    logging.disable(logging.NOTSET)
    logging.basicConfig(level=logging.DEBUG)
    test_agent = False
    test_perm = False
    test_huff = False

    # Testing Agent
    test_agent = True
    if test_agent:
        agent = Agent()
        msg = "abc"
        deck = agent.encode(msg)
        if not cards.valid_deck(deck):
            raise ValueError
        msg_decoded = agent.decode(deck)
        print(f"Message: {msg}")
        print(f"Shuffled deck: {deck}")
        print(f"Decoded Message: {msg_decoded}")

        # Check NULL redundancy
        count_null = 0
        for idx in range(1000):
            rng = np.random.default_rng(idx)
            deck_r = cards.generate_deck(rng, random=True)
            msg_r = agent.decode(deck_r)
            if msg_r == "NULL":
                count_null += 1

    # Testing the str-to-perm and back
    # test_perm = True
    if test_perm:
        encode_len = 12
        valid_cards = tuple(range(52 - encode_len, 52))
        test_perm = Perm(valid_cards, alpha)
        card_perm = test_perm.str_to_perm("hel loworld", max_chars=20)
        # card_perm = test_perm.str_to_perm("helloworld", max_chars=6)
        # card_perm = test_perm.str_to_perm("123456987", max)
        print("card seq = " + str(card_perm))
        decoded_str = test_perm.perm_to_str(card_perm)
        print("recovered string = " + decoded_str)

    # test_huff = True
    if test_huff:
        huff = Huffman()
        huff.print_codes()

        msg = "qqq"
        encoding = huff.encode(msg)
        print(f"Encoding - {msg}: {encoding}")
        decoded_msg = huff.decode(encoding)
        print(f"Decoding - {encoding}: {decoded_msg}")
        num = huff.encoding_to_num(encoding)
        print(f"Encoding as a num: {num}")

        # COMPARE HUFF VS DIRECT ENCODE
        encode_len = 12
        valid_cards = tuple(range(52 - encode_len, 52))
        test_perm = Perm(valid_cards, alpha)
        card_perm1 = test_perm.str_to_perm(msg, max_chars=20)
        decode1 = test_perm.perm_to_str(card_perm1)

        max_trials = 30
        while max_trials > 0:
            encoding = huff.encode(msg)
            num = huff.encoding_to_num(encoding)
            if test_perm.check_num_too_large(num):
                max_trials -= 1
                msg = msg[:-1]
                continue
            else:
                break

        if max_trials < 29:
            logging.warning(f"Huff: Message too large to encode. Shortening to : '{msg}'")
        card_perm2 = test_perm.num_to_perm(num)
        decode2 = test_perm.perm_to_num(card_perm2)
        decode2 = huff.num_to_binstr(decode2)
        decode2 = huff.decode(decode2)

        print(f"Direct Encode to Seq: {card_perm1}")
        print(f"Direct Decode: {decode1}")
        print(f"Huffma Encode to Seq: {card_perm2}")
        print(f"Huffma Decode: {decode2}")

    # Test Unscramble
    if True:
        deck = [45, 46, 47, 48]
        sdeck = [46, 45, 47, 48]
        perm_ck = Perm(tuple(deck), alpha)
        num = perm_ck.perm_to_num(deck)
        check_sum = calc_checksum(num, len(perm_ck.perm_zero))
        unscramble = Unscramble(sdeck, check_sum, perm_ck, max_trials=10000)
        cdeck = unscramble.unscramble()

        if cdeck == deck:
            print(f"Unscramble success")
        else:
            print(f"Unscramble Fail")
    print("Done")

