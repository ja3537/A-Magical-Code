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

vocab_paths = ['', 'messages/agent2/g2_vocab.txt', 'messages/agent2/g3_vocab.txt', '', 'messages/agent2/g5_vocab.txt', 'messages/agent2/g6_vocab.txt', 'messages/agent2/g7_vocab.txt', 'messages/agent2/g8_vocab.txt']

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
    elif group == 2:
        letter_freq = np.ones(26)
        letter_freq = letter_freq / letter_freq.sum() * (6/17)
        digit_freq = np.array([10,8,9,7,6,6,1,1,1,1])
        digit_freq = digit_freq / digit_freq.sum() * (9/17)
        space_freq = np.ones(1) * (2/17)
        freq = np.concatenate([letter_freq/100*95, digit_freq, space_freq]).tolist()
        ch = list(map(chr, range(97, 123)))
        chars = [word.upper() for word in ch] + list(map(str, range(10))) + [' ']
        freq_table = {c:f for c, f in zip(chars, freq)}
    else:
        return english_codec_w_digit()
    return HuffmanCodec.from_frequencies(freq_table) # 37 characters

def get_map(codec, mode, length, group):
    #print(group)
    with open(vocab_paths[group-1], 'r') as f:
        vocab = f.read().replace('\t', '').split('\n')

    # rank by bits needed to encode each combination
    chars = [c for c in list(codec.get_code_table().keys()) if type(c) is str] # exclude _EOF
    if group == 3 or group == 5:
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

# splits g3's password into list of words and digits in order
def split_password(encode_map, digits_and_words, curr, remaining_pw):
    if remaining_pw == '':
        if curr == '':
            return digits_and_words
        elif curr.isdigit() or curr in encode_map:
            return digits_and_words + [curr]
        else:
            return None

    if curr.isdigit() or curr in encode_map:
        extracted = split_password(encode_map, digits_and_words+[curr], '', remaining_pw)
        if not extracted is None:
            return extracted
    
    return split_password(encode_map, digits_and_words, curr+remaining_pw[0], remaining_pw[1:])

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
        self.checksum = 2**12 -1 #sum(range(53))
        self.n2 = -1

    def clean_text(self, s,group):
        truncated = False
        recognizable_chars = self.codec.get_code_table().keys()
        new_s = ''
        if group != 4 and group !=2:
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

    def truncate_and_encode(self, s, group):
        # truncate
        truncated = False
        max_perm = math.factorial(self.N_MAX)
        perm = float('inf')
        while perm > max_perm:
            if group == 2:
                encoded = self.codec.encode(s[:-4])
                encoded = self.add_checksum(encoded,group)
                perm = int.from_bytes(encoded, byteorder='big')
                #print(encoded)
                perm = self.add_year(perm, s[-4:])
            else:
                encoded = self.codec.encode(s)
                encoded = self.add_checksum(encoded,group)
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

    def add_year(self,perm,year):
        '''Use 2 bits to encode year'''
        #2023 -> 1
        #print(year[-1])
        return (perm << 2) + (int(year[-1]) - 2)
    
    def remove_year(self, perm):
        '''Remove last 2 bits'''
        year = perm - (perm >> 2 << 2)
        return perm >> 2, year + 2

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

    def add_checksum(self,message, group):
        d=hashlib.md5(message).digest(); d=base64.b64encode(d); 
        checksum = self.checksum - (sum(d))# // 10)
        
        checksum = (bin(checksum))
        g = bin(group-1)[2:]
        if len(g) == 1:
            g ='00' + g
        elif len(g) == 2:
            g = '0' + g
        checksum += g

        checksum = int(checksum,2)
        sb = checksum.to_bytes(2,"big")
        
        new_message = message + sb
        return new_message

    def encode_default(self, message, group):
        self.codec = get_codec(group)
        message, truncated = self.clean_text(message, group)
        partial = False
        partial |= truncated
        perm, truncated = self.truncate_and_encode(message, group)
        partial |= truncated
        return perm, partial

    def decode_default(self, b, group):
        #based on group get codec
        return self.codec.decode(b)


    def encode_w_vocab(self, message, group):
        length = 4 if group == 3 else 3 # 37^3 = 50653, 27^3 = 19683
        partial = False

        if group == 3:
            self.codec = english_codec_w_digit(letter_p=0.9, digit_p=0.09, space_p=0.01)
        else:
            self.codec = english_codec_w_digit()

        # map word in vocab to 3-char permutations
        encode_map = get_map(self.codec, mode='encode', length=length, group=group)

        if group == 3:
            words = split_password(encode_map, [], '', message[1:])
            #print(words)
            short_message = ''
            for w in words:
                if w.isdigit():
                    short_message += w
                elif w in encode_map:
                    short_message += encode_map[w]
                else:
                    partial = True

        elif group == 5:
            if message[-1] == ' ':
                message = message[:-1]
            else:
                partial = True
            words = message.split(' ')
            short_message = words[0]
            for w in [' '.join(words[1:-1]), words[-1]]:
                if w in encode_map:
                    short_message += encode_map[w]
                else:
                    partial = True
        else:
            words = message.split(' ')
            short_message = ''
            for w in words:
                if w in encode_map:
                    short_message += encode_map[w]
                else:
                    partial = True

        #print(short_message)
        perm, truncated = self.truncate_and_encode(short_message, group)
        partial |= truncated

        return perm, partial

    def decode_w_vocab(self, b, group, partial=False):
        length = 4 if group == 3 else 3
        short_message = self.codec.decode(b)
        if len(short_message) < 1:
            return 'NULL'
        #print(short_message)
        decode_map = get_map(self.codec, 'decode', length, group)
        if group == 3:
            s = '@'
            i, j = 0, 1
            while j <= len(short_message):
                if short_message[i:j].isdigit():
                    s += short_message[i:j]
                    i = j
                    j += 1
                elif j == i + length:
                    if short_message[i:j] in decode_map:
                        s += decode_map[short_message[i:j]]
                        i = j
                        j += 1
                    else:
                        partial = True
                        break
                else:
                    j += 1
            return s

        s = ''
        if group == 5:
            i = 0
            while i < len(short_message) and short_message[i].isdigit():
                s += short_message[i]
                i += 1
            short_message = short_message[i:] if i < len(short_message) else ''
            s += ' '
        #print(s, short_message)

        words = []
        for i in range(len(short_message) // length):
            mapped_word = short_message[length*i:length*i+length]
            if mapped_word in decode_map:
                words.append(decode_map[mapped_word])
            else:
                partial = True
                break
        s += ' '.join(words)

        if group == 5 and not partial:
            s += ' '

        return s

    def encode(self, message):
        group = 1
        split_message = message.split()
        if message[0] == "@":
            group = 3
        elif len(split_message) == 3 and len(split_message[0]) == 3 and (not split_message[0].isdigit()) and split_message[2].isdigit() and split_message[0] == split_message[0].upper():
            group = 2
        else:
            if len(split_message) > 1:
                if split_message[1] in ["N,","S,","E,","W,"]:
                    group = 4
                else:
                    group = 5
                    with open(vocab_paths[group-1], 'r') as f: 
                        vocab = f.read().replace('\t', '').split('\n')
                    if not (split_message[0].isdigit() and split_message[-1] in vocab):
                        group = 1

            if group == 1:
                group = 6
                with open(vocab_paths[group-1], 'r') as f: 
                    vocab = f.read().replace('\t', '').split('\n')
                for word in split_message:
                    if word not in vocab:
                        group = 1
                if group == 1:
                    group = 7
                    with open(vocab_paths[group-1], 'r') as f:
                        vocab = f.read().replace('\t', '').split('\n')
                    for word in split_message:
                        if word not in vocab:
                            group = 1
                if group == 1:
                    group = 8
                    with open(vocab_paths[group-1], 'r') as f:
                        vocab = f.read().replace('\t', '').split('\n')
                    for word in split_message:
                        if word not in vocab:
                            group = 1
                
            
        if group == 3 or group >= 5:
            perm, partial = self.encode_w_vocab(message, group=group)
        else:
            perm, partial = self.encode_default(message, group=group)
        choice = group

        # use 1 bit to encode partial
        perm = self.add_partial_flag(perm, partial)

        # use 3 bits to encode encoder choice
        perm = self.add_encoder_choice(perm, choice-1)
        ordered_deck = perm_decode(perm, self.N)
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

            perm = perm_encode(ordered_deck)
            if perm > 0:
                perm, choice = self.remove_encoder_choice(perm)
                choice += 1
                perm, partial = self.remove_partial_flag(perm)
                if choice == 2:
                    perm, year = self.remove_year(perm)
                byte_length = (max(perm.bit_length(), 1) + 7) // 8
                b = (perm).to_bytes(byte_length, byteorder='big')
                cs = int.from_bytes(b[-2:], byteorder='big')# >> 3
                
                if len(bin(cs)) > 4:
                    g = bin(cs)[-3:]
                    g = int(g,2) + 1
                cs = cs >> 3
                
                d=hashlib.md5(b[:-2]).digest(); d=base64.b64encode(d);  
                if cs == self.checksum - (sum(d)):# * 10):
                    passed_check = True
                else:
                    perm = -1
             
        if n_decode > N_MAX:
            msg = "NULL"
        else:
            group = choice
            if group == 3 or group >= 5:
                msg = self.decode_w_vocab(b[:-2], group=group, partial=partial)
            else:
                msg = self.decode_default(b[:-2], group=group)
            if partial and msg != 'NULL':
                msg  = 'PARTIAL: ' + msg
            else:
                if group == 2 and msg != 'NULL':
                    msg = msg + "202" + str(year)
        return msg


if __name__ == '__main__':
    print(generate(10, seedNum=2))