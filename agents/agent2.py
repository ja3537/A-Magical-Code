from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math


# def nthperm(l, n):
#     '''Get n-th permutation of l'''
#     l = list(l)

#     indices = []
#     for i in range(1, 1+len(l)):
#         indices.append(n % i)
#         n //= i
#     indices.reverse()

#     perm = []
#     for index in indices:
#         # Using pop is kind of inefficient. We could probably avoid it.
#         perm.append(l.pop(index))
#     return tuple(perm)

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
        self.codec = load_shakespeare()
        self.N = 20 # only modify bottom N cards

    def encode(self, message):
        encoded = self.codec.encode(message)
        perm = int.from_bytes(encoded, byteorder='big')
        ordered_deck = perm_decode(perm, self.N) # perm may be larger than N; need to change later
        deck = list(range(52-self.N)) + [card+(52-self.N) for card in ordered_deck]
        return deck


    def decode(self, deck):
        # TODO: incorporate shuffling
        
        ordered_deck = [card-(52-self.N) for card in deck[-self.N:]]

        # hardcoded check for a random deck
        if len(set(ordered_deck).intersection(set(range(self.N)))) < 10:
            return 'NULL'

        # decode last N cards
        perm = perm_encode(ordered_deck)
        byte_length = (max(perm.bit_length(), 1) + 7) // 8
        b = (perm).to_bytes(byte_length, byteorder='big')
        msg = self.codec.decode(b)
        print(msg)
        return msg