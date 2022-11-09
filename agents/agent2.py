from tkinter import E
from dahuffman import HuffmanCodec
from dahuffman import load_shakespeare
import sys
import math
import numpy as np
import re, pickle
import copy, itertools
import hashlib, base64
from random import *
import datetime
import secrets
import string

############# GENERATOR ###############
def generate(numMessages, seedNum):
        seed(seedNum)
        messages = []
        for m in range(numMessages):
            file = open('messages/agent2/airportcodes.txt', 'r')
            content = file.readlines()
            month = randint(1,12)
            day = randint(1,28)
            airport = randint(1,2019)
            airportCode = content[airport]
            airportCode = airportCode[:-1]
            if month < 10:
                month = '0' + str(month)
            if day < 10:
                day = '0' + str(day)
            N = 4
            res = ''.join(choice(string.ascii_uppercase + string.digits)
                        for i in range(N))
            message = airportCode + ' ' + res + ' ' + str(month) + str(day) + '2023' 
            messages.append(message)
        return messages



vocab_paths = ['', '', 'messages/agent2/g3_vocab.txt', '', '', 'messages/agent2/g6_vocab.txt', 'messages/agent2/g7_vocab.txt', 'messages/agent2/g8_vocab.txt']

def english_codec_w_digit(letter_p=0.92, digit_p=0.03, space_p=0.05):
    # https://en.wikipedia.org/wiki/Letter_frequency
    letter_freq = np.array([8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.36, 0.15, 1.974, 0.074])
    letter_freq = letter_freq / letter_freq.sum() * letter_p # reserve 92% for letters
    digit_freq = np.ones(10) / letter_freq.sum() * digit_p # reserve 3% for digits
    space_freq = np.ones(1) * space_p # reserve 5% for space
    freq = np.concatenate([letter_freq/100*95, digit_freq, space_freq]).tolist()
    chars = list(map(chr, range(97, 123))) + list(map(str, range(10))) + [' ']
    freq_table = {c:f for c, f in zip(chars, freq)}
    return HuffmanCodec.from_frequencies(freq_table) # 37 characters

def get_codec(group):
    if group == 4:
        freq_table = {'N':1, 'S':1, 'W':1, 'E':1, ' ':3, ',':2, '.':4,
                '1':5, '2':5, '3':5, '4':5, '5':5, '6':5, '7':5, '8':5, '9':5, '0':5}
    else:
        return english_codec_w_digit()
    return HuffmanCodec.from_frequencies(freq_table) # 37 characters

def get_map(codec, mode, length, group):
        with open(vocab_paths[group-1], 'r') as f:
            vocab = f.read().replace('\t', '').split('\n')

        # rank by bits needed to encode each combination
        chars = [c for c in list(codec.get_code_table().keys()) if type(c) is str] # exclude _EOF
        if group == 3:
            chars = [c for c in chars if not c.isdigit()] # exclude digits
        code_table = codec.get_code_table()
        all_combi = [(combi, sum([code_table[c][0] for c in combi])) for combi in itertools.combinations_with_replacement(chars, length)]
        ranked_combi = [combi for combi, _ in sorted(all_combi, key=lambda x:x[1])]

        # map word in vocab to 3-char permutations
        str_map = {}
        i = 0
        for combi in ranked_combi:
            target_strs = sorted(list(set([''.join(p) for p in itertools.permutations(combi)])))
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
        self.N_MAX = 30
        self.checksum = 2**16 -1 #sum(range(53))
        self.n2 = -1

    def generator(self, numMessages, seedNum):
        seed(seedNum)
        messages = []
        for m in range(numMessages):
            file = open('messages/agent2/airportcodes.txt', 'r')
            content = file.readlines()
            month = randint(1,12)
            day = randint(1,28)
            airport = randint(1,2019)
            airportCode = content[airport]
            airportCode = airportCode[:-1]
            if month < 10:
                month = '0' + str(month)
            if day < 10:
                day = '0' + str(day)
            N = 4
            res = ''.join(choice(string.ascii_uppercase + string.digits)
                        for i in range(N))
            message = airportCode + ' ' + res + ' ' + str(month) + str(day) + '2023' 
            messages.append(message)
        return messages

    def clean_text(self, s,group):
        truncated = False
        recognizable_chars = self.codec.get_code_table().keys()
        new_s = ''
        if group != 4:
            s = re.sub('\s\s+', ' ', s.lower())
        else:
            s = re.sub('\s\s+', ' ', s)
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

        #print(s)
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
        #print(message)
       # m = bytes(message,'utf-8')
        d=hashlib.md5(message).digest(); d=base64.b64encode(d);  
        checksum = self.checksum - sum(d)
        #print(checksum)
        sb = checksum.to_bytes(2,"big")
        new_message = message + sb
        return new_message

    def encode_default(self, message, group):
        if group != 4:
            self.codec = english_codec_w_digit() 
            message, truncated = self.clean_text(message, group)
        else:
            self.codec = get_codec(group)
            message, truncated = self.clean_text(message, group)
        partial = False
        partial |= truncated
        perm, truncated = self.truncate_and_encode(message)
        partial |= truncated
        return perm, partial

    def decode_default(self, b, group):
        #based on group get codec
        return self.codec.decode(b)

    def encode_w_vocab(self, message, group):
        length = 3 # 37^3 = 50653
        partial = False

        if group == 3:
            self.codec = english_codec_w_digit(letter_p=0.9, digit_p=0.09, space_p=0.01)
        else:
            self.codec = english_codec_w_digit()

        # map word in vocab to 3-char permutations
        encode_map = get_map(self.codec, mode='encode', length=length, group=group)

        if group == 3:
            s = message[1:]
            short_message = ''
            i, j = 0, 1
            while j <= len(s):
                if s[j-1:j].isdigit():
                    if i != j - 1: 
                        partial = True
                    short_message += s[j-1:j]
                    i = j
                    j += 1
                # this has higher scores:
                # if s[i:j].isdigit():
                #     short_message += s[j-1:j]
                #     i = j
                #     j += 1
                elif s[i:j] in encode_map:
                    short_message += encode_map[s[i:j]]
                    i = j
                    j += 1
                else:
                    j += 1
                #print(i, j, short_message)
            if i < len(message[1:]):
                partial = True

        else:
            words = message.split(' ') if group == 8 else message.lower().split(' ')
            short_message = ''

            for w in words:
                if w in encode_map:
                    short_message += encode_map[w]
                else:
                    partial = True

        #print(short_message)
        perm, truncated = self.truncate_and_encode(short_message)
        partial |= truncated

        return perm, partial

    def decode_w_vocab(self, b, group):
        short_message = self.codec.decode(b)
        #print(short_message)
        decode_map = get_map(self.codec, mode='decode', length=3, group=group)
        if group == 3:
            s = '@'
            i, j = 0, 1
            while j <= len(short_message):
                if short_message[j-1:j].isdigit():
                    s += short_message[i:j]
                    i = j
                    j += 1
                # this has higher scores:
                # if short_message[i:j].isdigit():
                #     s += short_message[i:j]
                #     i = j
                #     j += 1
                elif j == i + 3:
                    s += decode_map[short_message[i:j]]
                    i = j
                    j += 1
                else:
                    j += 1
                #print(i, j, s)

        else:
            words = []
            for i in range(len(short_message) // 3):
                mapped_word = short_message[3*i:3*i+3]
                if mapped_word in decode_map:
                    words.append(decode_map[mapped_word])
            s = ' '.join(words)

        return s

    def encode(self, message):
        # TODO: select encoder with the smallest perm
        group = 1
        # attempting to encode based on structure
        #should encode every group but 5 currently
        #still need to get shortest when applicable
        split_message = message.split()
        #print(split_message)
        if message[0] == "@":
            group = 3
        elif len(split_message[0]) == 3:
            group = 2
        else:
            if len(split_message) > 1:
                if split_message[1] in ["N,","S,","E,","W,"]:
                    group = 4
            if group == 1:
                group = 6
                with open(vocab_paths[group-1], 'r') as f: #group 6
                    vocab = f.read().replace('\t', '').split('\n')
                for word in split_message:
                    if word not in vocab:
                        group = 1
                if group == 1:
                    group = 7
                    with open(vocab_paths[group-1], 'r') as f: #group 6
                        vocab = f.read().replace('\t', '').split('\n')
                    for word in split_message:
                        if word not in vocab:
                            group = 1
                if group == 1:
                    group = 8
                    with open(vocab_paths[group-1], 'r') as f: #group 6
                        vocab = f.read().replace('\t', '').split('\n')
                    for word in split_message:
                        if word not in vocab:
                            group = 1
                
            
        #print("encode group: " + str(group))
        #group = 7
        if group == 3 or group >= 6:
            perm, partial = self.encode_w_vocab(message, group=group)
        else:
            perm, partial = self.encode_default(message, group=group)
        
        choice = group

        # use 1 bit to encode partial
        perm = self.add_partial_flag(perm, partial)

        # use 3 bits to encode encoder choice
        perm = self.add_encoder_choice(perm, choice)
        #print(perm)
        ordered_deck = perm_decode(perm, self.N)
        #print(self.N, ordered_deck)
        self.start = 52 - self.N 
        #print(self.N)
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
                d=hashlib.md5(b[:-2]).digest(); d=base64.b64encode(d);  
                if sum(d) == self.checksum - cs:
                    passed_check = True
                # if sum(b[:-2]) + cs == self.checksum:
                #     print(sum(b[:-2]) + cs)
                #     passed_check = True
                #     #print("ACCEPT")
                else:
                    perm = -1
             
            #print(perm,n_decode,not(passed_check))

        # b'\xc4N\xb1\xc7\x19\xc4\xc7RK4\x92\xcd8\xf9i'
        # [14, 21, 25, 0, 15, 5, 32, 22, 29, 26, 16, 6, 19, 30, 31, 9, 23, 20, 27, 8, 12, 3, 18, 7, 24, 10, 28, 17, 1, 4, 13, 2, 11]
        
        #print("DECODE group: " + str(choice))
        #print(n_decode)
        if n_decode > N_MAX:
            #print(n_decode)
            msg = "NULL"
        else:
            # TODO: select decoder
            group = choice
            #group =7 
            if group == 3 or group >= 6:
                msg = self.decode_w_vocab(b[:-2], group=group)
            else:
                msg = self.decode_default(b[:-2], group=group)
            if partial:
                msg  = 'PARTIAL: ' + msg
        #print(msg)
        return msg


if __name__ == '__main__':
    print(generate(10, seedNum=2))