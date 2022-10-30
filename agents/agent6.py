import numpy as np
from itertools import permutations
import nltk
from nltk.stem import SnowballStemmer
from nltk import LancasterStemmer
from nltk.tag import pos_tag
import string
from decimal import *
import math

#nltk.download('averaged_perceptron_tagger')

#HYPERPARAMETERS FOR ARITHMATIC AGENT
#first SAFE_CARDS amount of cards, we do not care about
#eg. we dont care about cards 0,1,2, ...., SAFE_CARDS-1
CARDS_FOR_ARITHMETIC_CODING = 26
PADDING_CARDS = 52 - CARDS_FOR_ARITHMETIC_CODING
ARITH_ACCURACY = 26

############################################################################
############################# HELPER FUNCTIONS #############################
############################################################################

#from Group 4
def cards_to_number(cards):
    num_cards = len(cards)

    if num_cards == 1:
        return 0

    ordered_cards = sorted(cards)
    permutations = math.factorial(num_cards)
    sub_list_size = permutations // num_cards
    sub_list_indx = sub_list_size * ordered_cards.index(cards[0])

    return int(sub_list_indx) + int(cards_to_number(cards[1:]))

#from Group 4
def number_to_cards(number, current_deck):
    num_cards = len(current_deck)

    if num_cards == 1:
        return current_deck

    ordered_cards = sorted(current_deck)
    permutations = math.factorial(num_cards)
    sub_list_size = permutations // num_cards
    sub_list_indx = int(Decimal(number) / sub_list_size)
    sub_list_start = sub_list_indx * sub_list_size

    if sub_list_start >= permutations:
        raise Exception('Number too large to encode in cards.')

    first_card = ordered_cards[sub_list_indx]
    ordered_cards.remove(first_card)
    return [first_card, *number_to_cards(int(number - sub_list_start), ordered_cards)]

#######################################################################
############################# AGENT CODES #############################
#######################################################################

class ArtihmaticCodingAgent:
    def __init__(self):

        self.values = " "+string.ascii_lowercase

        #index of permutations
        #self.perm_idx = list(itertools.permutations(list(range(52-CARDS_FOR_ARITHMETIC_CODING, 52))))

        #arithmetic coding based on these frequencies
        #https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
        self.arithmatic_freq = {
            "e": 11.1607,
            "a": 8.4966,
            "r": 7.5809,
            "i": 7.5448,
            "o": 7.1635,
            "t": 6.9509,
            "n": 6.6544,
            "s": 5.7351,
            "l": 5.4893,
            "c": 4.5388,
            "u": 3.6308,
            "d": 3.3844,
            "p": 3.1671,
            "m": 3.0129,
            "h": 3.0034,
            "g": 2.4705,
            "b": 2.0720,
            "f": 1.8121,
            "y": 1.7779,
            "w": 1.2899,
            "k": 1.1016,
            "v": 1.0074,
            "x": 0.2902,
            "z": 0.2722,
            "j": 0.1000,
            "q": 0.1000,
            " ": 0.0961,
            "$": 0.0965
        }

        self.arith_boundaries = {}

        total = 0
        prev = 0
        for c in self.arithmatic_freq:
            total += self.arithmatic_freq[c]/100.0
            self.arith_boundaries[c] = (prev, total if total <= 1 else 1)
            prev = total
        #print(self.arith_boundaries)

        getcontext().prec = 50

    def get_arithmatic_code(self, message):

        #currently only supports lowercase letters
        message = message.lower()

        min_bound = Decimal(0)
        max_bound = Decimal(1)

        for c in message+"$":
            small, big = self.arith_boundaries[c]

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small)*r
            max_bound = min_bound + Decimal(big-small)*r

        val = (max_bound+min_bound)/2
        return val

    def get_word(self, decimal_value):
        result = ""
        while len(result) < 30:
            for c in self.arith_boundaries:
                min_bound = Decimal(self.arith_boundaries[c][0])
                max_bound = Decimal(self.arith_boundaries[c][1])

                if decimal_value > min_bound and decimal_value < max_bound:
                    if c == "$":
                        return result
                    result += c
                    decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                elif decimal_value == min_bound or decimal_value == max_bound:
                    raise Exception("Error in Parsing Word")
        return result

    def encode(self, message):

        val = Decimal(self.get_arithmatic_code(message))
        
        #convert decimal to binary
        val_as_int = int(str(val)[2:2+ARITH_ACCURACY])

        #encode to a card sequence
        padded_cards = list(range(0,PADDING_CARDS))
        arith_cards = list(range(PADDING_CARDS, 52))

        encoded_cards = number_to_cards(val_as_int, arith_cards)

        #add padded cards to the end
        return padded_cards+encoded_cards

    def decode(self, deck):
        
        #Get order of last 26
        encoded_cards = []
        for num in deck:
            if num >= PADDING_CARDS:
                encoded_cards.append(num)
        #find the decimal value from it
        val = int(cards_to_number(encoded_cards))
        val_as_Decimal = Decimal("0."+"0"*(ARITH_ACCURACY-len(str(val)))+str(val))

        #TODO IDEAS:
        #first card is the whether the number of 1's is even or odd
        #check the rest of the deck to confirm the number
        #check the delimiter at the end

        return self.get_word(val_as_Decimal)

class HauffmanAgent:
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

###################################################################################
###################################################################################
###################################################################################

class Agent:
    def __init__(self):
        #change this when needed
        self.agent = ArtihmaticCodingAgent()

    def encode(self, message):
        return self.agent.encode(message)

    def decode(self, deck):
        return self.agent.decode(deck)

###################################################################################
###################################################################################
###################################################################################


if __name__ == "__main__":
    agent = Agent()
    message = "indonesian flag"
    deck = agent.encode(message)
    print(agent.decode(deck))
