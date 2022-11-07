from tkinter import E
from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math
import numpy as np
import re, pickle
import copy, itertools

vocab_paths = ['', '', '', '', '', 'messages/agent2/g6_vocab.txt', 'messages/agent2/g7_vocab.txt', 'messages/agent2/g8_vocab.txt']

def english_codec_w_digit():
    # https://en.wikipedia.org/wiki/Letter_frequency
    letter_freq = np.array([8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.36, 0.15, 1.974, 0.074])
    letter_freq = letter_freq / letter_freq.sum() * 0.92 # reserve 92% for letters
    digit_freq = np.ones(10) / letter_freq.sum() * 0.03 # reserve 3% for digits
    space_freq = np.ones(1) * 0.05 # reserve 5% for space
    freq = np.concatenate([letter_freq/100*95, digit_freq, space_freq]).tolist()
    chars = list(map(chr, range(97, 123))) + list(map(str, range(10))) + [' ']
    freq_table = {c:f for c, f in zip(chars, freq)}
    return HuffmanCodec.from_frequencies(freq_table) # 37 characters

def get_map(codec, mode, length, group):
        with open(vocab_paths[group-1], 'r') as f:
            vocab = f.read().replace('\t', '').split('\n')

        # rank by bits needed to encode each combination
        chars = list(codec.get_code_table().keys())[1:]
        code_table = codec.get_code_table()
        all_combi = [(combi, sum([code_table[c][0] for c in combi])) for combi in itertools.combinations_with_replacement(chars, length)]
        ranked_combi = [combi for combi, _ in sorted(all_combi, key=lambda x:x[1])]

        # map word in vocab to 3-char permutations
        str_map = {}
        i = 0
        for combi in ranked_combi:
            target_strs = set([''.join(p) for p in itertools.permutations(combi)])
            for target in target_strs:
                if mode == 'encode':
                    str_map[vocab[i]] = target
                else:
                    str_map[target] = vocab[i]
                i += 1
                if i >= len(vocab): break
            if i >= len(vocab): break

        return str_map

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
    A = []
    for i in range(n-1,-1,-1):
        cards_left = n-i
        value,pos = divmod(value, cards_left)
        A.insert(pos,i)
    return A


class Agent:
    def __init__(self):
        #self.codec = english_codec_w_digit()
        #self.codec.print_code_table()
        self.N_MAX = 30
        self.checksum = 2**16 -1 #sum(range(53))
        self.n2 = -1

    def clean_text(self, s):
        truncated = False
        recognizable_chars = self.codec.get_code_table().keys()
        new_s = ''
        s = re.sub('\s\s+', ' ', s.lower())
        s = s.replace('\t', ' ')
        for c in s:
            if c in recognizable_chars:
                new_s += c
            else:
                truncated = True
        return new_s, truncated

    def retrieve_coded_cards(self, deck, n_decode):
        cards_for_encoding = set(range(n_decode))

        cards = []
        for c in [card-(52-n_decode) for card in deck]:
            if c in cards_for_encoding:
                cards.append(c)

        return cards

    def truncate_and_encode(self, s):
        # truncate
        truncated = False
        max_perm = math.factorial(self.N_MAX)
        perm = float('inf')
        while perm > max_perm:
            encoded = self.codec.encode(s)
            encoded = self.add_checksum(encoded)
            perm = int.from_bytes(encoded, byteorder='big')
            perm = self.add_partial_flag(perm)
            perm = self.add_encoder_choice(perm) # IMPORTANT: add placeholder bits for length calculation
            if perm > max_perm:
                s = s[:-1] 
                truncated = True

        print(s)
        N = 4
        while math.factorial(N) <= perm:
            N += 1
        self.N = N

        perm, _ = self.remove_encoder_choice(perm)
        perm, _ = self.remove_partial_flag(perm)

        return perm, truncated

    def add_partial_flag(self, perm, partial=False):
        '''Add one bit to the end of byte'''
        return (perm << 1) + int(partial)

    def remove_partial_flag(self, perm):
        '''Remove last bit'''
        partial = bool(perm - (perm >> 1 << 1))
        return perm >> 1, partial

    def add_encoder_choice(self, perm, choice=0):
        '''Use 3 bits to encode encoder'''
        return (perm << 3) + choice

    def remove_encoder_choice(self, perm):
        '''Remove and use last 3 bits to determine the encoder'''
        choice = perm - (perm >> 3 << 3)
        return perm >> 3, choice

    def add_checksum(self,message):
        checksum = self.checksum - sum(message)
        sb = checksum.to_bytes(2,"big")
        new_message = message + sb
        return new_message

    def encode_default(self, message):
        self.codec = english_codec_w_digit()
        partial = False
        message, truncated = self.clean_text(message)
        partial |= truncated
        perm, truncated = self.truncate_and_encode(message)
        partial |= truncated
        return perm, partial

    def decode_default(self, b):
        return self.codec.decode(b)

    def encode_w_vocab(self, message, group):
        length = 3 # 37^3 = 50653
        partial = False

        self.codec = english_codec_w_digit()

        # map word in vocab to 3-char permutations
        encode_map = get_map(self.codec, mode='encode', length=length, group=group)
        em = encode_map

        words = message.split(' ') if group == 8 else message.lower().split(' ')
        short_message = ''

        for w in words:
            if w in encode_map:
                short_message += encode_map[w]
            else:
                partial = True

        perm, truncated = self.truncate_and_encode(short_message)
        partial |= truncated

        return perm, partial

    def decode_w_vocab(self, b, group):
        short_message = self.codec.decode(b)
        print(short_message)
        decode_map = get_map(self.codec, mode='decode', length=3, group=group)
        words = []
        for i in range(len(short_message) // 3):
            mapped_word = short_message[3*i:3*i+3]
            if mapped_word in decode_map:
                words.append(decode_map[mapped_word])
        return ' '.join(words)

    def encode(self, message):
        # TODO: select encoder with the smallest perm
        group = 6
        perm, partial = self.encode_w_vocab(message, group=group)
        choice = 1

        # use 1 bit to encode partial
        perm = self.add_partial_flag(perm, partial)

        # use 3 bits to encode encoder choice
        perm = self.add_encoder_choice(perm, choice)

        ordered_deck = perm_decode(perm, self.N)
        #print(self.N, ordered_deck)
        self.start = 52 - self.N 
        deck = list(range(self.start)) + [card+self.start for card in ordered_deck]
        return deck

    def decode(self, deck):
        N_MAX = self.N_MAX 
        n_decode = 2
        perm = -1
        passed_check = False
        
        while perm <= 0 and n_decode <= N_MAX and not(passed_check):

            n_decode += 1 # do not put this after retrieve_coded_cards... N_MAX would be wrong
            ordered_deck = self.retrieve_coded_cards(deck, n_decode)
            #print(n_decode, ordered_deck)
            #if n_decode == N_MAX: print(ordered_deck)

            perm = perm_encode(ordered_deck)
            #print("perm: " + str(perm))

            if perm > 0:
                perm, choice = self.remove_encoder_choice(perm)
                perm, partial = self.remove_partial_flag(perm)
                byte_length = (max(perm.bit_length(), 1) + 7) // 8
                b = (perm).to_bytes(byte_length, byteorder='big')
                cs = int.from_bytes(b[-2:], byteorder='big')
                if sum(b[:-2]) + cs == self.checksum:
                    passed_check = True
                    #print("ACCEPT")
                else:
                    perm = -1
            
            #print(perm,n_decode,not(passed_check))

        # b'\xc4N\xb1\xc7\x19\xc4\xc7RK4\x92\xcd8\xf9i'
        # [14, 21, 25, 0, 15, 5, 32, 22, 29, 26, 16, 6, 19, 30, 31, 9, 23, 20, 27, 8, 12, 3, 18, 7, 24, 10, 28, 17, 1, 4, 13, 2, 11]
        if n_decode > N_MAX:
            #print(n_decode)
            msg = "NULL"
        else:
            # TODO: select decoder
            group = 6
            msg = self.decode_w_vocab(b[:-2], group=group)
            if partial:
                msg  = 'PARTIAL: ' + msg
        return msg