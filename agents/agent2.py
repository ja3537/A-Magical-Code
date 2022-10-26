from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math


class Agent:
    def __init__(self):
        self.codec = load_shakespeare()

    def encode(self, message):
        deck = []
        encoded = self.codec.encode(message)
        
        permutation = float(bin(int.from_bytes(encoded, byteorder=sys.byteorder))[2:])
        n = 32
        value = permutation
        for i in range(n-1,-1,-1):
            cards_left = n-i
            value,pos = divmod(value, cards_left)
            deck.insert(pos,i)
        for i in range(52):
            if i not in deck:
                deck = [i] + deck
        return deck


    def decode(self, deck):
        #figure out how many times shuffled...
        #decode last 30 cards
        value = 0
        n = 32
        A = deck[20:] # Take a copy to avoid modifying original input array 
        for i in range(n):
            cards_left = n-i
            pos = A.index(i)
            del A[pos]
            value = value * cards_left + pos
        b = (value).to_bytes(2, byteorder='big')
        msg = self.codec.decode(b)
        #change calue to bytes
        #self.codec.decode()
        return msg