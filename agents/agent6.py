from cards import generate_deck
import numpy as np
from itertools import permutations

class Agent:
    def __init__(self):
        self.huff_LtoB = {
            'a': '1111',
            'b': '100000',
            'c': '00000',
            'd': '11101',
            'e': '110',
            'f': '00011',
            'g': '100001',
            'h': '0111',
            'i': '0110',
            'j': '0001010111',
            'k': '0001011',
            'l': '11100',
            'm': '00001',
            'n': '1001',
            'o': '1011',
            'p': '01000',
            'q': '000101010',
            'r': '1010',
            's': '0101',
            't': '001',
            'u': '01001',
            'v': '000100',
            'w': '100011',
            'x': '00010100',
            'y': '100010',
            'z': '0001010110'
        }
        self.huff_BtoL = {}
        for l in self.huff_LtoB:
            binary = self.huff_LtoB[l]
            self.huff_BtoL[binary] = l

        self.permu_map ={}
        all_permu = permutations([1, 2, 3, 4])
        for i in range(24):
            self.permu_map[i]=all_permu[i]

    def numberToBase(self,n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]

    def baseToNumber(self, arr, b):
        if len(arr) == 0:
            return 0
        power = len(arr) - 1
        num = 0
        for a in arr:
            num += a * (b ** power)
            power -= 1
        return num

    def countLeadingZeros(self,binary):
        num = 0
        cur = 0
        while binary[cur] != '1':
            num += 1
            cur += 1
        return num

    def encodeDeck(self,encode_num, num_leading_zero):
        return

    def decodeDeck(self,deck):
        return


    def encode(self, message):
        deck = list(range(52))

        print("message:", message)
        binary = ''
        for letter in message:
            huffman_code = self.huff_LtoB[letter]
            binary += huffman_code

        num_leading_zero = self.countLeadingZeros(binary)

        # print("encode binary:", binary)
        # print("leading zero:",num_leading_zero)
        decimal = int(binary, 2)
        print("first decimal:", decimal)
        encode_num = self.numberToBase(decimal, 24)

        print(encode_num)
        #return encode_num, num_leading_zero
        deck=self.encodeDeck(encode_num, num_leading_zero)

        return deck

    def decode(self, deck):
        encode_num,num_leading_zero=self.decodeDeck(deck)

        decimal_num = self.baseToNumber(encode_num, 24)
        binary_arr = self.numberToBase(decimal_num, 2)
        binary_str = ''.join([str(x) for x in binary_arr])

        add_zero = '0' * num_leading_zero
        final_binary = add_zero + binary_str
        # print(final_binary)

        ans = []
        i = 0
        j = 0
        while j <= len(final_binary):
            if final_binary[i:j] in self.huff_BtoL:
                ans.append(self.huff_BtoL[final_binary[i:j]])
                i = j
            else:
                j += 1

        decode_massage = ''.join(ans)
        print("decode:", decode_massage)

        return decode_massage