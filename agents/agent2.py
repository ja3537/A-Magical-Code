from tkinter import E
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
    if len(A) == 0:
        return -1
    value = 0
    n = len(A)
    A = A[:] # Take a copy to avoid modifying original input array 
    for i in range(n):
       cards_left = n-i
       try:
        pos = A.index(i)
       except:
        return -1
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
        self.N = 25 # only modify bottom N cards
        self.start, self.end = 52-self.N, 51 # for locating reserved cards
        self.checksum = 2**16 -1 #sum(range(53))
        self.n2 = -1

    def clean_text(self, s):
        recognizable_chars = self.codec.get_code_table().keys()
        new_s = ''
        for c in s.lower():
            new_s += (c if c in recognizable_chars else '')
        return new_s

    def retrieve_coded_cards(self, deck,n_decode):
       # start_i = deck.index(self.start)
       # end_i = deck.index(self.end)
        no_check = True
        
        start = 52 - n_decode
        end = 51
        start_i = deck.index(start)
        end_i = deck.index(end)
        deck_window = [c-(start+1) for c in deck[start_i+1:end_i]]
        # print("d"+str(deck_window))
        cards_for_encoding = set(range(n_decode-2))

        cards = []
        for c in deck_window:
            if c in cards_for_encoding:
                cards.append(c)
       # print("cards" +str(cards))

        # if start_i+1 >= end_i:
        #     # encoding's messed up
        #     return []

        # deck_window = [c-(self.start+1) for c in deck[start_i+1:end_i]]
        # cards_for_encoding = set(range(self.N-2))

        # cards = []
        # for c in deck_window:
        #     if c in cards_for_encoding:
        #         cards.append(c)

        # if len(cards) != self.N-2:
        #     return []

        return cards

    def truncate_and_encode(self, s):
    #     max_perm = math.factorial(self.N-2)
    #     perm = float('inf')
    #     while perm > max_perm:
    #         encoded = self.codec.encode(s)
    #         perm = int.from_bytes(encoded, byteorder='big')
    #         s = s[:-1]

        #encode based on N:
        


        encoded = self.codec.encode(s)
        encoded = self.add_checksum(encoded)
        #print("ENCODED" + str(encoded))
        perm = int.from_bytes(encoded, byteorder='big')
        N = 2
        while math.factorial(N-2) < perm:
            N += 1

        #print(N)
        #print(N)
        self.N = N
        self.start, self.end = 52-self.N, 51
        
        # reduce N if we can (not allowed); TODO: encode N with cards 48-52
        # while perm < math.factorial(self.N-2):
        #     self.N -= 1
        # self.N += 1
        # self.start, self.end = 52-self.N, 51

        return perm

    def add_checksum(self,message):
        checksum = self.checksum - sum(message)
        #print("cs" + str(checksum))
        sb = checksum.to_bytes(2,"big")
        new_message = message + sb
        #print(sb)
        return new_message

    def encode(self, message):
       # print("message: " + message)
        message = self.clean_text(message)
        #print("message: " + message)
        perm = self.truncate_and_encode(message)
        #print("trunc and encode: " + str(perm))
        ordered_deck = perm_decode(perm, self.N-2) # perm may be larger than N!; need to change later
        #print("ordered deck: " + str(ordered_deck))
        deck = list(range(52-self.N)) + [self.start] + [card+(self.start+1) for card in ordered_deck] + [self.end]
        #print("deck: " + str(deck))
        return deck

    def decode(self, deck):
        n_decode = 10
        perm = -1
        passed_check = False
        N_MAX = 40
        
        while perm <= 0 and n_decode < N_MAX and not(passed_check):
            #print(n_decode)
            ordered_deck = self.retrieve_coded_cards(deck,n_decode)
            #print()
            #print()
            #print("retrieved_deck: " + str(ordered_deck))
            #if len(ordered_deck) == 0:
            #    return 'NULL'

            # decode last N cards

            perm = perm_encode(ordered_deck)
            #print("perm: " + str(perm))
            n_decode += 1
            if perm > 0:
                byte_length = (max(perm.bit_length(), 1) + 7) // 8
                b = (perm).to_bytes(byte_length, byteorder='big')
                # print(b)
                # print(sum(b))
                # print(n_decode)
                # print(b[-2:])
                cs = int.from_bytes(b[-2:], byteorder='big')
                if sum(b[:-2]) + cs == self.checksum:
                    passed_check = True
                    print("ACCEPT")
                else:
                    perm = -1
            print(perm,n_decode,not(passed_check))
       # b'\xc4N\xb1\xc7\x19\xc4\xc7RK4\x92\xcd8\xf9i'
       # [14, 21, 25, 0, 15, 5, 32, 22, 29, 26, 16, 6, 19, 30, 31, 9, 23, 20, 27, 8, 12, 3, 18, 7, 24, 10, 28, 17, 1, 4, 13, 2, 11]
        if n_decode >= N_MAX:
            msg = "NULL"
        else:
            msg = self.codec.decode(b[:-2])
        return msg