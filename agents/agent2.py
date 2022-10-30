from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math
import numpy as np

def english_codec_w_digit():
    # https://en.wikipedia.org/wiki/Letter_frequency
    freq = np.array([8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.36, 0.15, 1.974, 0.074])
    freq = np.concatenate([freq/100*99, np.ones(10)/10]).tolist()
    chars = list(map(chr, range(97, 123))) + list(map(str, range(10)))
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
        self.N = 20 # only modify bottom N cards

    def encode(self, message):
        encoded = self.codec.encode(message)
        perm = int.from_bytes(encoded, byteorder='big')
        ordered_deck = perm_decode(perm, self.N) # perm may be larger than N; need to change later
        deck = list(range(52-self.N)) + [card+(52-self.N) for card in ordered_deck]
        return deck


    def decode(self, deck):
        ordered_deck = [card-(52-self.N) for card in deck[-self.N:]]

        # hardcoded check for a random deck
        lost_cards = list(set(range(self.N)) - set(ordered_deck))
        if len(lost_cards) >= self.N / 2:
            return 'NULL'

        # replace inserted cards (remove error when shuffling)
        temp_deck = []
        insert_i = 0
        for card in ordered_deck:
            if not card in set(range(self.N)):
                temp_deck.append(lost_cards[insert_i])
                insert_i += 1
            else:
               temp_deck.append(card) 
        ordered_deck = temp_deck

        # decode last N cards
        perm = perm_encode(ordered_deck)
        byte_length = (max(perm.bit_length(), 1) + 7) // 8
        b = (perm).to_bytes(byte_length, byteorder='big')
        msg = self.codec.decode(b)
        return msg