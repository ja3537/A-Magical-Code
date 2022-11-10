import math
import os
import hashlib

UNTOK = '*'
EMPTY = ''
DICT_SIZE = 27000
SENTENCE_LEN = 6
CHECKSUM_CARDS = 6

class EncoderDecoder:
    def __init__(self, n):
        self.encoding_len = n
        if n < 7: #If less than 7 bits its for checksum
            self.perm_zero = [46,47,48,49,50,51]
        else:
            self.perm_zero = list(range(46-n, 46)) #[20,21,...45]
        self.max_messge_length = 12 #TODO TEST AND CHANGE THIS VALUE
        factorials = [0] * n
        for i in range(n):
            factorials[i] = math.factorial(n-i-1)
        self.factorials = factorials

        words_dict = {EMPTY: 0}
        words_index = [EMPTY]
        with open(os.path.dirname(__file__) + '/../messages/agent7/30k.txt') as f:
            for i in range(1, DICT_SIZE-1):     # reserve 1 for empty and 1 for unknown
                word = f.readline().rstrip('\t\n')
                words_dict[word] = i
                words_index.append(word)
        words_dict[UNTOK] = DICT_SIZE-1
        words_index.append(UNTOK)
        self.words_dict = words_dict
        self.words_index = words_index

    def perm_number(self, permutation):
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

    def nth_perm(self, n):
        perm = []
        items = self.perm_zero[:]

        for f in self.factorials:
            lehmer = n // f
            x = items.pop(lehmer)
            perm.append(x)
            n %= f
        return perm

    def str_to_perm(self, message):
        tokens = message.split()
        init = [EMPTY for i in range(SENTENCE_LEN)]
        for i in range(len(tokens)):
            init[i] = tokens[i]
        tokens = init[::-1]
        num = 0
        for i in range(SENTENCE_LEN):
            num += self.words_dict.get(tokens[i], DICT_SIZE-1) * DICT_SIZE**i
        return self.nth_perm(num)

    def str_to_num(self, message):
        tokens = message.split()
        init = [EMPTY for i in range(SENTENCE_LEN)]
        for i in range(len(tokens)):
            init[i] = tokens[i]
        tokens = init[::-1]
        num = 0
        for i in range(SENTENCE_LEN):
            num += self.words_dict.get(tokens[i], DICT_SIZE-1) * DICT_SIZE**i
        return num

    def perm_to_str(self, perm):
        num = self.perm_number(perm)
        words = []
        for i in range(SENTENCE_LEN):
            index = num % DICT_SIZE
            words.append(self.words_index[index])
            num = num // DICT_SIZE
        return ' '.join(words[::-1]).strip()

    def set_checksum(self, num, base=10):
        num_bin = bin(num)[2:]
        chunk_len = 5
        checksum = 0
        mod_prime = 113
        while len(num_bin) > 0:
            bin_chunk = num_bin[:chunk_len]
            num_bin = num_bin[chunk_len:]

            num_chunk = int(bin_chunk, 2)
            checksum = ((checksum + num_chunk) * base) % mod_prime
        return checksum


class Agent:
    def __init__(self, encoding_len=26):
        self.encoding_len = encoding_len

        self.ed = EncoderDecoder(self.encoding_len)
        self.perm_ck = EncoderDecoder(6)

    def encode(self, message):
        print('Encoding "', message, '"')

        x  = self.ed.str_to_num(message)
        checksum = self.ed.set_checksum(x)
        checksum_cards = self.perm_ck.nth_perm(checksum)
        a = list(range(46 - self.encoding_len))
        b = self.ed.str_to_perm(message)
        c =checksum_cards
        encoded_deck = a+b+c
        print('Encoded deck:\n', encoded_deck, '\n---------')
        return encoded_deck

    def decode(self, deck):
        msg_perm = []
        checksum = []
        for card in deck:
            if 20 <= card <= 45:
                msg_perm.append(card)
            if card > 45:
                checksum.append(card)

        #print('\nMessage Cards:', msg_perm)
        #print('Checksum Cards:', checksum)
        msg_num = self.ed.perm_number(msg_perm)

        decoded_checksum = self.perm_ck.perm_number(checksum)
        message_checksum = self.perm_ck.set_checksum(msg_num)        

        #print(decoded_checksum)
        #print(message_checksum)
        if message_checksum != decoded_checksum:
            #print("MESSAGE CHECKSUM IS NOT EQUAL TO DECODED CHECKSUM")
            return "NULL"
        else:
            return self.ed.perm_to_str(msg_perm)