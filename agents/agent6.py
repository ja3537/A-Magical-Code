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

STOP_SYMBOL = "ß"
ARITH_START = 0

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

        getcontext().prec = 100

        #index:
        #0 : lower Alphabet
        #1 : symbols
        #2 : number
        #3 : base
        #4 : Capital Alphabet

        #arithmetic coding based on these frequencies
        #https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.alphabet_freq = {
            "e": 0.111607,
            "a": 0.084966,
            "r": 0.075809,
            "i": 0.075448,
            "o": 0.071635,
            "t": 0.069509,
            "n": 0.066544,
            "s": 0.057351,
            "l": 0.054893,
            "c": 0.045388,
            "u": 0.036308,
            "d": 0.033844,
            "p": 0.031671,
            "m": 0.030129,
            "h": 0.030034,
            "g": 0.024705,
            "b": 0.020720,
            "f": 0.018121,
            "y": 0.017779,
            "w": 0.012899,
            "k": 0.011016,
            "v": 0.010074,
            "x": 0.002902,
            "z": 0.002722,
            "j": 0.001965,
            "q": 0.001961,
        }

        self.base_freq = {
            " ": 0.5,
            ".": 0.3,
            STOP_SYMBOL: 0.2
        }

        self.alpha_caps = {}
        self.caps= "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in self.caps:
            self.alpha_caps[i] = 1/len(self.caps)

        #all Symbol/Number Freq
        #unused symbols: None
        self.num= "0123456789"
        self.num_freq = {}
        for i in self.num:
            self.num_freq[i] = 1/len(self.num)

        self.symbols = "!#$%&'*,-/?@[]_`()<=>;:^{|}~\\+"
        self.symbol_freq = {}
        for i in self.symbols:
            self.symbol_freq[i] = 1/len(self.symbols)


        #TODO: create script to finalize proportions of encoding
        #Group 1: lowercase alphabet + base + ","
        #Group 2: capital alphabet + number + " "
        #Group 3: passwords: lowercase letters + numbers + (no spaces)

        #USE Special Method
        #Group 4: number (always 4 decimals) + north/south + west/east
            #EG. 23.5504 S, 46.6339 W

        #Group 5: number + Lowercase Alphabet + base + " -#' "
            #Add encoding because all words start with uppercase

        #Group 6: lowercase Alphabet + base
        #Group 7: longer words + coordinates (?)
        #Group 8: lowercase Alphabet + base
            #EG. Names + Places, all have capital first character

        #General Encoder + for each group 
        #sort by length of messages (where shortest messages should be the highest weight)
        self.freq_d = [self.alphabet_freq, self.symbol_freq, self.num_freq, self.base_freq, self.alpha_caps]
        self.weight_dict = {
            1: [0.4, 0.3, 0.199, 0.001, 0.1], #alphabetical + number + symbols + alphabeticalCaps + base
            2: [0.499, 0.499, 0.002, 0], #
            3: [0.999, 0, 0, 0.001, 0], #only alphabetical + base
            4: [0, 0, 0.001, 0.999], #only alphabeticalCaps
            5: [0, 0.999, 0.001, 0], #Numerical + Symbols + base
            6: [0, 0.7, 0.1, 0.2] # For Group4: Numbers + Base + Capital
        }

    def change_frequencies(self, freq, maximum):

        total = 0
        for v in freq.values():
            total += v

        prop = maximum/total

        d = {}
        for key,value in freq.items():
            d[key] = value*prop

        return d

    def get_boundaries_based_on_lead_number(self, number):

        weights = self.weight_dict[int(number)]

        d = {}

        for i in range(4):
            if weights[i] > 0:
                d.update(self.change_frequencies(self.freq_d[i], weights[i]))
        
        return d


    def set_arithmatic_boundaries(self, arith_freq):
        maximum = 0
        for v in arith_freq.values():
            maximum += v

        proportion = Decimal((maximum-ARITH_START)/1)

        total = prev = Decimal(ARITH_START)
        arith_boundaries = {}
        for c in arith_freq:
            val = Decimal(arith_freq[c])*proportion
            total += val
            arith_boundaries[c] = (prev, total if total <= 1 else 1)
            prev = total

        #print(arith_boundaries)
        return arith_boundaries

    def get_arithmatic_code(self, message, arith_boundaries):

        min_bound = Decimal(0)
        max_bound = Decimal(1)

        for c in message+STOP_SYMBOL:
            small, big = arith_boundaries[c]

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small)*r
            max_bound = min_bound + Decimal(big-small)*r

        #Removes all extraneous digits eg. If 0.555030434 has same value as 0.55, will shorten to 0.55
        str_min = str(min_bound)
        str_max = str(max_bound)

        val = ""

        for i in range(len(str_max)):
            val += str_max[i]
            if str_max[i] != str_min[i]:
                break

        return val

    def get_word(self, decimal_value, arith_boundaries):
        #print(decimal_value)
        result = ""
        while len(result) < 30:
            check = True
            for c in arith_boundaries:
                min_bound = Decimal(arith_boundaries[c][0])
                max_bound = Decimal(arith_boundaries[c][1])

                if decimal_value > min_bound and decimal_value < max_bound:
                    result += c
                    if c == STOP_SYMBOL:
                        return result
                    decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                    check = False
                elif decimal_value == min_bound or decimal_value == max_bound:
                    return "NULL"

            if check:
                return "NULL"

        return result

    def encode(self, message):
        #print(message)
       

        word_set = set(message)
        min_val = float("inf")

        #try multiple length encodings
        for i in self.weight_dict.keys():
            #print(i)
            curr_dict = self.get_boundaries_based_on_lead_number(i)

            #check if a particular encoding has the given subsets
            char_set = set(curr_dict.keys())
            #print(char_set)
            if word_set.issubset(char_set):
                val = Decimal(self.get_arithmatic_code(message,self.set_arithmatic_boundaries(curr_dict)))
                val_as_int = int(str(i)+str(val)[2:])
                #print(val_as_int)

                min_val = min(min_val, val_as_int)

        if min_val == float("inf"):
            return "NULL"

        #encode to a card sequence
        deck = list(range(0,52))
        
        #padded_cards = list(range(0,PADDING_CARDS))
        #arith_cards = list(range(PADDING_CARDS, 52))
        #print(min_val)
        encoded_deck = number_to_cards(min_val, deck)

        #add padded cards to the end
        #print(encoded_deck)
        return encoded_deck

    def decode_helper(self, threshhold_value, deck):

        #take cards given threshhold value
        encoded_cards = []
        for num in deck:
            if num >= 51-threshhold_value:
                encoded_cards.append(num)
 
        #find the decimal value from it
        val = int(cards_to_number(encoded_cards))
        #print(val)

        number_in_front = int(str(val)[0])
        

        if number_in_front not in self.weight_dict.keys():
            return "NULL"

        else:

            #take the int as a Decimal
            val_as_Decimal = Decimal("0."+str(val)[1:])
            #print(val_as_Decimal)
            #print(self.set_arithmatic_boundaries((self.get_boundaries_based_on_lead_number(number_in_front))))

            return self.get_word(val_as_Decimal,self.set_arithmatic_boundaries((self.get_boundaries_based_on_lead_number(number_in_front))))


    def decode(self, deck):
        #print("Decoding")

        word_count = {}
        max_word_count = 0
        max_word = None

        #try all encoding lengths
        for i in range(3, 52):
            #print(i)
            word = self.decode_helper(i, deck)  
            #print(word)

            #word needs to have STOP signal
            if word != "NULL" and STOP_SYMBOL in word:
                word_count[word] = word_count[word] + 1 if word in word_count else 1

                #Choose the best word (should appear at least once times and has a stop signal)
                if word_count[word] > max_word_count:
                    max_word_count = word_count[word]
                    max_word = word
        #print(word_count)
        
        if max_word in word_count and word_count[max_word] > 1:
            return max_word[:-1]
        else:
            return "NULL"
        #return max_word[:-1] if max_word is not None else "NULL"


########################################################################################################

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
    agent = ArtihmaticCodingAgent()
    
    message = "hello i am maximo oen. bye bye"
    deck = agent.encode(message)
    print(deck)
    print(agent.decode(deck))
