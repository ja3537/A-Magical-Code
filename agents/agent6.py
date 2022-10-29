from cards import generate_deck
import numpy as np
from itertools import permutations
import nltk
from nltk.stem import SnowballStemmer
from nltk import LancasterStemmer
from nltk.tag import pos_tag

nltk.download('averaged_perceptron_tagger')

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
        all_permu = list(permutations([1, 2, 3, 4]))
        for i in range(24):
            self.permu_map[i] = all_permu[i]

        # Snowball is less aggressive meaning less risk for loss of meaning but larger output
        self.sno = SnowballStemmer('english')
        # Lancaster is the most aggressive stemmer
        self.lanc = LancasterStemmer()

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

    # Condenses message using stemming
    def condenseMessage(self, message_string):
        tagged_sentence = pos_tag(message_string.split())
        reduced_string = ""

        for word,tag in tagged_sentence:
            # Do not stem proper nouns
            if tag != 'NNP':
                reduced_string += self.lanc.stem(word) + " "
            else:
                reduced_string += word + " "
        return reduced_string.strip()

    def countLeadingZeros(self,binary):
        num = 0
        cur = 0
        while binary[cur] != '1':
            num += 1
            cur += 1
        return num

    def encodeDeck(self,encode_num, num_leading_zero):
        encode_num.reverse()
        cur = 12
        used_deck = []
        for num in encode_num:
            permu = self.permu_map[num]
            group_start_card = cur * 4
            group = {1: group_start_card, 2: group_start_card + 1, 3: group_start_card + 2, 4: group_start_card + 3}
            permu_card = [group[p] for p in permu]
            used_deck.append(permu_card)
            cur -= 1

        permu_zero = self.permu_map[num_leading_zero]

        group_zero = {1: 0, 2: 1, 3: 2, 4: 3}
        permu_card = [group_zero[p] for p in permu_zero]
        used_deck.append(permu_card)
        used_deck.reverse()

        deck = []
        for i in range(1, cur + 1):
            group_start_card = i * 4
            deck.append(group_start_card)
            deck.append(group_start_card + 1)
            deck.append(group_start_card + 2)
            deck.append(group_start_card + 3)

        for group in used_deck:
            for card in group:
                deck.append(card)

        print(deck)
        return deck

    def decodeDeck(self, deck):
        count_unused = 0
        start_card = 0
        while (deck[start_card] != 0):
            start_card += 1
            count_unused += 1

        start_group = [0, 1, 2, 3]
        expected_num_unsused = max(deck[:start_card])

        used_deck = []

        for card in deck[start_card:]:
            if card in start_group or card > expected_num_unsused:
                used_deck.append(card)

        #print(used_deck)
        decode_permu = []
        i = 0
        while i < len(used_deck):
            group = (used_deck[i] % 4 + 1, used_deck[i + 1] % 4 + 1, used_deck[i + 2] % 4 + 1, used_deck[i + 3] % 4 + 1)
            print(group)
            i += 4
            for p in self.permu_map:
                if self.permu_map[p] == group:
                    decode_permu.append(p)

        num_leading_zero = decode_permu.pop(0)
        print(num_leading_zero)
        print(decode_permu)

        return num_leading_zero,decode_permu

    def encode(self, message):
        print("message:", message)
        reduced_message = self.condenseMessage(message)
        print("reduced message by "+str(len(message)-len(reduced_message))+" chars, new message:" , reduced_message)

        binary = ''
        for letter in reduced_message:
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
        num_leading_zero,encode_num=self.decodeDeck(deck)

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