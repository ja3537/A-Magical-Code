from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math
import numpy as np

def english_codec_w_digit():
    # TODO: special characters?
    # https://en.wikipedia.org/wiki/Letter_frequency
    letter_freq = np.array([8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.36, 0.15, 1.974, 0.074])
    letter_freq = letter_freq / letter_freq.sum() * 0.92 # reserve 92% for lettes
    digit_freq = np.ones(10) / letter_freq.sum() * 0.03 # reserve 3% for digits
    space_freq = np.ones(1) * 0.05 # reserve 5% for space
    freq = np.concatenate([letter_freq/100*95, digit_freq, space_freq]).tolist()
    chars = list(map(chr, range(97, 123))) + list(map(str, range(10))) + [' ']
    freq_table = {c:f for c, f in zip(chars, freq)}
    return HuffmanCodec.from_frequencies(freq_table)

def perm_encode(A):
    value = 0
    n = len(A)
    A = A[:] # Take a copy to avoid modifying original input array 
    for i in range(n):
       cards_left = n-i
       pos = A.index(i)
       del A[pos]
       value = value * cards_left + pos
    return value

def perm_decode(value, n):
    A=[]
    for i in range(n-1,-1,-1):
       cards_left = n-i
       value,pos = divmod(value, cards_left)
       A.insert(pos,i)
    return A


class Agent:
    def __init__(self):
        self.codec = english_codec_w_digit()
        #self.codec.print_code_table()
        self.N = 20 # only modify bottom N cards
        self.start, self.end = 52-self.N, 51 # for locating reserved cards

    def clean_text(self, s):
        recognizable_chars = self.codec.get_code_table().keys()
        new_s = ''
        for c in s.lower():
            new_s += (c if c in recognizable_chars else '')
        return new_s

    def retrieve_coded_cards(self, deck):
        start_i = deck.index(self.start)
        end_i = deck.index(self.end)
        if start_i+1 >= end_i:
            # encoding's messed up
            return []
        deck_window = [c-(self.start+1) for c in deck[start_i+1:end_i]]
        cards_for_encoding = set(range(self.N-2))

        cards = []
        for c in deck_window:
            if c in cards_for_encoding:
                cards.append(c)

        if len(cards) != self.N-2:
            return []

        return cards

    def truncate_and_encode(self, s):
        max_perm = math.factorial(self.N-2)
        perm = float('inf')
        while perm > max_perm:
            encoded = self.codec.encode(s)
            perm = int.from_bytes(encoded, byteorder='big')
            s = s[:-1]

        # reduce N if we can
        while perm < math.factorial(self.N-2):
            self.N -= 1
        self.N += 1
        self.start, self.end = 52-self.N, 51

        return perm

    def encode(self, message):
        message = self.clean_text(message)
        perm = self.truncate_and_encode(message)
        ordered_deck = perm_decode(perm, self.N-2) # perm may be larger than N!; need to change later
        deck = list(range(52-self.N)) + [self.start] + [card+(self.start+1) for card in ordered_deck] + [self.end]
        return deck

    def decode(self, deck):
        ordered_deck = self.retrieve_coded_cards(deck)
        if len(ordered_deck) == 0:
            return 'NULL'

        # decode last N cards
        perm = perm_encode(ordered_deck)
        byte_length = (max(perm.bit_length(), 1) + 7) // 8
        b = (perm).to_bytes(byte_length, byteorder='big')
        msg = self.codec.decode(b)
        return msg