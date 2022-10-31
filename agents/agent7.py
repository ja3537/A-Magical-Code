import math
import os

UNTOK = '*'
EMPTY = ''
DICT_SIZE = 27000
SENTENCE_LEN = 6


class EncoderDecoder:
    def __init__(self, n=26):
        self.encoding_len = n
        # characters = " 1234567890abcdefghijklmnopqrstuvwxyz"
        # self.char_dict, self.bin_dict = self.binary_encoding_dicts(characters)
        self.perm_zero = list(range(50-n, 50))
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

    # @staticmethod
    # def to_binary_padded(n):
    #     ret = ''
    #     while n > 1:
    #         ret += str(n % 2)
    #         n = n // 2
    #     ret += str(n)
    #     return '0' * (6 - len(ret)) + ret[::-1]
    #
    # @staticmethod
    # def to_binary(n):
    #     ret = ''
    #     while n > 1:
    #         ret += str(n % 2)
    #         n = n // 2
    #     ret += str(n)
    #     return ret[::-1]
    #
    # def binary_encoding_dicts(self, characters):
    #     char_dict = {}
    #     bin_dict = {}
    #     for i in range(len(characters)):
    #         b = self.to_binary_padded(i)
    #         char_dict[characters[i]] = b
    #         bin_dict[b] = characters[i]
    #
    #     return char_dict, bin_dict

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
            perm.append(items.pop(lehmer))
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

    def perm_to_str(self, perm):
        num = self.perm_number(perm)
        words = []
        for i in range(SENTENCE_LEN):
            index = num % DICT_SIZE
            words.append(self.words_index[index])
            num = num // DICT_SIZE
        return ' '.join(words[::-1]).strip()


    # def str_to_perm(self, s):
    #     ret = ''
    #     for c in s[:14]:
    #         ret += self.char_dict[c]
    #     n = int(ret, 2)
    #     return self.nth_perm(n)
    #
    # def perm_to_str(self, permutation):
    #     n = self.perm_number(permutation)
    #     binary_string = self.to_binary(n)
    #     binary_string = '0' * ((6 - len(binary_string) % 6) % 6) + binary_string
    #
    #     ret = ''
    #     for i in range(0, len(binary_string) - 5, 6):
    #         if binary_string[i:i + 6] not in self.bin_dict:
    #             return 'PARTIAL: '
    #         ret += self.bin_dict[binary_string[i:i + 6]]
    #     return ret


class Agent:
    def __init__(self, encoding_len=26):
        self.encoding_len = encoding_len
        self.ed = EncoderDecoder(self.encoding_len)

    def encode(self, message):
        return list(range(50 - self.encoding_len)) + self.ed.str_to_perm(message)[::-1] + [50, 51]

    def decode(self, deck):
        perm = []
        for card in deck:
            if 24 <= card <= 51:
                perm.append(card)
        perm = perm[:-2][::-1] + perm[-2:]

        print(perm)
        if perm[-2:] != [50, 51]:
            return "NULL"
        # if perm[:2] != [22, 23]:
        #     return "PARTIAL:"

        return self.ed.perm_to_str(perm[:-2])

# ed = EncoderDecoder(26)
# p = ed.str_to_perm('')
# s = ed.perm_to_str(p)
#
# print(p)
# print(f'#{s}#')