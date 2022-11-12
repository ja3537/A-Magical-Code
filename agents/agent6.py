import numpy as np
from itertools import permutations
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

# Check for groups in order how simplistic and sure it is to check
def identify_domain(message):

    message = message.strip()

    # check for group 3
    if message[0] == '@':
        return 3

    message_split = message.split(" ")
    # check for group 2
    if len(message_split) == 3:
        if len(message_split[0]) == 3 and len(message_split[1]) == 4 and len(message_split[2]) == 8:
            return 2

    # check for group 4
    if len(message_split) == 4 and message.count(".") == 2:
        # Double check
        if (len(message_split[0]) >= 6 and len(message_split[0]) <= 8) and (len(message_split[2]) >= 6 and len(message_split[2]) <= 8) and len(message_split[1]) == 2 and len(message_split[3]) == 1:
            return 4

    # check for group 5
    if message_split[0].isdigit():
        # Fairly small file
        with open("../messages/agent5/street_suffix.txt") as file_in:
            lines = []
            for line in file_in:
                lines.append(line.rstrip())
            #print(message_split[-1])
            if message_split[-1] in lines:
                return 5

    # check for group 6 // Need to check how expensive this is should be better because we arent loading it all into memory at once
    with open("../messages/agent6/unedited_corpus.txt") as file_in:
        lines = []
        for line in file_in:
            lines.append(line.rstrip())

        if message in lines:
            return 6

    # At this point the only corpus with numbers is group 1 
    if any(char.isdigit() for char in message):
        return 1

    # check for group 8
    # All agent 8 are greater than 1 in length and less than 6
    if len(message_split) > 1 and len(message_split) < 6:
        names = []
        places = []
        with open("../messages/agent8/names.txt") as file_in:
            for line in file_in:
                names.append(line.rstrip())

        with open("../messages/agent8/places.txt") as file_in:
            for line in file_in:
                places.append(line.rstrip())

        if all(word in places or word in names for word in message_split):
            return 8

    # Check group 7 possibly the largest file
    lines = []
    with open("../messages/agent7/30k.txt") as file_in:
        for line in file_in:
            lines.append(line.rstrip())

        if all(word in lines for word in message_split):
            return 7

    # Check if message should be group 1
    if len(message) >2 and len(message) < 13:
        return 1

    #unidentifiable domain
    return 0

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

        """self.values = " "+string.ascii_lowercase"""

        #index of permutations
        #self.perm_idx = list(itertools.permutations(list(range(52-CARDS_FOR_ARITHMETIC_CODING, 52))))

        getcontext().prec = 200

        #index:
        #0 : lower Alphabet
        #1 : symbols
        #2 : number
        #3 : base
        #4 : Capital Alphabet

        #arithmetic coding based on these frequencies
        #https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
        """self.alphabet = "abcdefghijklmnopqrstuvwxyz"
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
            self.symbol_freq[i] = 1/len(self.symbols)"""


        #TODO: create script to finalize proportions of encoding
        #Group 1: lowercase alphabet + base + ","
        #Group 2: capital alphabet + number + " "
        #Group 3: passwords: lowercase letters + numbers + (no spaces)
            #same amount of digits as words

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
        """self.freq_d = [self.alphabet_freq, self.symbol_freq, self.num_freq, self.base_freq, self.alpha_caps]
        self.weight_dict = {
            1: [0.4, 0.3, 0.199, 0.001, 0.1], #alphabetical + number + symbols + alphabeticalCaps + base
            #2: [0.499, 0.499, 0.002, 0], #
            3: [0.999, 0, 0, 0.001, 0], #only alphabetical + base
            4: [0, 0, 0.001, 0.999], #only alphabeticalCaps
            5: [0, 0.999, 0.001, 0], #Numerical + Symbols + base
            6: [0, 0.7, 0.1, 0.2] # For Group4: Numbers + Base + Capital
        }
"""

        self.domain2_airport_codes_to_num = {}
        self.domain2_num_to_airport_codes = {}
        with open("../messages/agent2/airportcodes.txt") as file_in:
            line_num = 0
            for line in file_in:
                self.domain2_airport_codes_to_num[line.rstrip()] = line_num
                self.domain2_num_to_airport_codes[line_num] = line.rstrip()
                line_num += 1
        
        self.domain5_stname_to_num = {}
        self.domain5_num_to_stname = {}
        with open("../messages/agent5/street_name.txt") as file_in:
            line_num = 0
            for line in file_in:
                if line.rstrip() not in self.domain5_stname_to_num:
                    self.domain5_stname_to_num[line.rstrip()] = line_num
                    self.domain5_num_to_stname[line_num] = line.rstrip()
                    line_num = len(self.domain5_stname_to_num)+1
        
        self.domain5_stsuffix_to_num = {}
        self.domain5_num_to_stsuffix = {}
        with open("../messages/agent5/street_suffix.txt") as file_in:
            line_num = 0
            for line in file_in:
                if line.rstrip() not in self.domain5_stsuffix_to_num:
                    self.domain5_stsuffix_to_num[line.rstrip()] = line_num
                    self.domain5_num_to_stsuffix[line_num] = line.rstrip()
                    line_num = len(self.domain5_stsuffix_to_num)+1

        #domain 3
        self.domain3_freq = {}
        with open("../messages/agent3/dicts/large_cleaned_long_words.txt") as file_in:
            line_num = 0
            for line in file_in:
                self.domain3_freq[line.rstrip()] = 1

        for i in range(0,10):
            self.domain3_freq[str(i)] = 1

        self.domain3_boundaries = self.set_arithmatic_boundaries(self.domain3_freq)

        #domain 8
        self.domain8_freq = {}
        with open("../messages/agent8/names.txt") as file_in:
            for line in file_in:
                self.domain8_freq[line.rstrip()] = 1

        with open("../messages/agent8/places.txt") as file_in:
            for line in file_in:
                self.domain8_freq[line.rstrip()] = 1

        self.domain8_freq[STOP_SYMBOL] = 1
        self.domain8_boundaries = self.set_arithmatic_boundaries(self.domain8_freq)
        #print(self.domain8_boundaries)

        #domain 7
        self.domain7_freq = {}
        with open("../messages/agent7/30k.txt") as file_in:
            for line in file_in:
                self.domain7_freq[line.rstrip()] = 1

        self.domain7_freq[STOP_SYMBOL] = 1
        self.domain7_boundaries = self.set_arithmatic_boundaries(self.domain7_freq)

        #domain 6
        self.domain6_freq = {}
        with open("../messages/agent6/unedited_corpus.txt") as file_in:
            for line in file_in:
                for word in line.split(" "):
                    self.domain6_freq[word.rstrip()] = 1

        self.domain6_freq[STOP_SYMBOL] = 1
        self.domain6_boundaries = self.set_arithmatic_boundaries(self.domain7_freq)


        #domain 2
        self.domain2_flight_details = {}
        for i in string.ascii_uppercase + string.digits:
            self.domain2_flight_details[i] = 1

        self.domain2_flight_boundaries = self.set_arithmatic_boundaries(self.domain2_flight_details)

        #domain 1
        self.domain1_freq = " 0123456789abcdefghijklmnopqrstuvwxyz."
        self.domain1_freq = {}
        for i in " 0123456789abcdefghijklmnopqrstuvwxyz.":
            self.domain1_freq[i] = 1/len(" 0123456789abcdefghijklmnopqrstuvwxyz.")
        
        self.domain1_boundaries = self.set_arithmatic_boundaries(self.domain1_freq)

        print("Finished initialization of dictionaries....")

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

        proportion = Decimal(1/maximum)

        total = prev = Decimal(ARITH_START)
        arith_boundaries = {}
        for c in arith_freq:
            val = Decimal(arith_freq[c])*proportion
            total += val
            arith_boundaries[c] = (prev, total if total <= 1 else 1)
            prev = total

        #print(arith_boundaries)
        return arith_boundaries

    def get_arithmatic_code(self, message, arith_boundaries, stop_symbol=STOP_SYMBOL, domain_2 = False):

        min_bound = Decimal(0)
        max_bound = Decimal(1)

        if stop_symbol != None:
            if type(message) == str:
                message += stop_symbol
            else:
                message.append(stop_symbol)

        for c in message:
            small, big = arith_boundaries[c]
            #print(small)
            #print(big)

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small)*r
            max_bound = min_bound + Decimal(big-small)*r

        #Removes all extraneous digits eg. If 0.555030434 has same value as 0.55, will shorten to 0.55
        str_min = str(min_bound)
        str_max = str(max_bound)

        #print(str_min)
        #print(str_max)

        val = ""
        for i in range(len(str_max)):
            val += str_max[i]
            if str_max[i] != str_min[i]:
                break

        if domain_2:
            return min_bound, max_bound

        return val

    def get_word(self, decimal_value, arith_boundaries, message_length, delimiter=" ", domain_3 = False):
        if domain_3:
            message_length -= 1

        #print(decimal_value)
        result = ""
        while len(result) < message_length:
            check = True

            for c in arith_boundaries:
                min_bound = Decimal(arith_boundaries[c][0])
                max_bound = Decimal(arith_boundaries[c][1])

                if decimal_value > min_bound and decimal_value < max_bound:
                    result += c

                    if len(result) == message_length:
                        if domain_3:
                            return "@"+result
                        return result
                    else:
                        result += delimiter

                    #if c == STOP_SYMBOL:
                    #    return result
                    decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                    check = False
                    break

                elif decimal_value == min_bound or decimal_value == max_bound:
                    return "NULL"

            if check:
                return "NULL"

        #if len(result) != message_length:
        #    return "NULL"

        if domain_3:
            return "@"+result

        return result

    def get_word_domain4(self, decimal_value, message_length, directions, delimiter=" "):
        #print(decimal_value)
        result = ""
        idx = 0


        while idx < 2:
            #get the first number, 
            if idx == 0:

                bin = Decimal(1)/Decimal(900001)
                bin = int(decimal_value // bin)

                for c in range(bin-4, bin+4):
                    min_bound = Decimal(c)/Decimal(900001)
                    max_bound = Decimal(c+1)/Decimal(900001)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        #print(c)
                        string_c = len(str(c))
                        value = str(c/10000)

                        while len(value.split(".")[1]) < 4:
                            value += "0"

                        result += value
                        result += " "+directions[0]+", "
                        

                        #if c == STOP_SYMBOL:
                        #    return result
                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"

            #get longitude
            elif idx == 1:
                bin = Decimal(1)/Decimal(1800001)
                bin = int(decimal_value // bin)
                for c in range(bin-4, bin+4):
                    min_bound = Decimal(c)/Decimal(1800001)
                    max_bound = Decimal(c+1)/Decimal(1800001)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        #print(c)
                        string_c = len(str(c))
                        value = str(c/10000)
                        
                        while len(value.split(".")[1]) < 4:
                            value += "0"

                        result += value
                        result += " "+directions[1]

                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"

            idx += 1
            #print(result)
            if check:
                return "NULL"

        #print(result)
        #print(len(result))
        #if len(result) != message_length:
            #print("Wrong Length")
        #   return "NULL"

        return result

    def get_word_domain5(self, decimal_value, message_length, delimiter=" "):
        #print(decimal_value)
        result = ""
        idx = 0
        while idx < 3:
            #get the first number
            if idx == 0:
                for c in range(10000):
                    min_bound = Decimal(c)/Decimal(10000)
                    max_bound = Decimal(c+1)/Decimal(10000)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        result += str(c)
                        result += " "

                        #if c == STOP_SYMBOL:
                        #    return result
                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"

            #get street name
            elif idx == 1:
                for c in range(len(self.domain5_stname_to_num)+1):
                    min_bound = Decimal(c)/Decimal(len(self.domain5_stname_to_num)+1)
                    max_bound = Decimal(c+1)/Decimal(len(self.domain5_stname_to_num)+1)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        result += self.domain5_num_to_stname[c]
                        result += delimiter

                        #if c == STOP_SYMBOL:
                        #    return result
                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"
            
            elif idx == 2:
                for c in range(len(self.domain5_stsuffix_to_num)+1):
                    min_bound = Decimal(c)/Decimal(len(self.domain5_stsuffix_to_num)+1)
                    max_bound = Decimal(c+1)/Decimal(len(self.domain5_stsuffix_to_num)+1)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        result += self.domain5_num_to_stsuffix[c]
                        result += " "

                        #if c == STOP_SYMBOL:
                        #    return result
                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"
            idx += 1
            if check:
                return "NULL"

        #print((result))
        if message_length > len(result):
            if message_length - len(result) <= 4:
                return "0"*(message_length - len(result)) + result

        if len(result)-message_length == 1:
            return result[:-1]

        #if len(result) != message_length:
            #print("Wrong Length")
        #    return "NULL"

        return result

    def get_word_domain2(self, decimal_value, message_length):
        #print(decimal_value)
        result = ""
        idx = 0
        while idx < 6:
            #get the first number
            if idx == 0:
                for c in range(2020):
                    min_bound = Decimal(c)/Decimal(2020)
                    max_bound = Decimal(c+1)/Decimal(2020)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        result += self.domain2_num_to_airport_codes[c]
                        result += " "

                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"

            #get street name
            elif idx >= 1 and idx <= 4 :
                for c in self.domain2_flight_boundaries:
                    min_bound = self.domain2_flight_boundaries[c][0]
                    max_bound = self.domain2_flight_boundaries[c][1]

                    if decimal_value > min_bound and decimal_value < max_bound:
                        result += c

                        #if c == STOP_SYMBOL:
                        #    return result
                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"
            
            elif idx == 5:
                for c in range(1008):
                    min_bound = Decimal(c)/Decimal(1008)
                    max_bound = Decimal(c+1)/Decimal(1008)

                    if decimal_value > min_bound and decimal_value < max_bound:
                        result += " "

                        year = str(2023+c//(12*28))
                        month = str((c%(12*28))//28+1)
                        date = str((c%(12*28))%28+1)

                        if len(month) < 2:
                            month = "0"+month

                        if len(date) < 2:
                            date = "0"+date

                        result += month+date+year

                        #if c == STOP_SYMBOL:
                        #    return result
                        decimal_value = Decimal((decimal_value-min_bound) / (max_bound-min_bound))
                        check = False
                        break

                    elif decimal_value == min_bound or decimal_value == max_bound:
                        return "NULL"

            idx += 1
            if check:
                return "NULL"
        #print(result)
        return result

    def encode_helper(self, message, domain_number):
        
        #domain 1 - equivalent spacing for each number
        if domain_number == 1:
            val = Decimal(self.get_arithmatic_code(message, self.domain1_boundaries, stop_symbol=None))
            return str(val)[2:]

        elif domain_number == 2:
            
            message_list = message.split(" ")

            #set dictionary
            airport_code = message_list[0]
            ac_code_num = self.domain2_airport_codes_to_num[airport_code]

            #random
            flight_details = message_list[1]

            #date
            date = message_list[2]

            #change date to value after the min date which is 01012023 : 0
            #min = 2023*336 + 01*28 + 01*28
            #year = 12*28 = 336
            #max value : 3 years * 12 months * 28 days = 1008
            
            month = int(date[:2])
            day = int(date[2:4])
            year = int(date[4:])
            day_diff = (year*336 + (month-1)*28 + day - 1) - 2023*336

            #airport codes
            min_bound = Decimal(0)
            max_bound = Decimal(1)

            small_ac = Decimal(ac_code_num)/Decimal(2020)
            big_ac = Decimal(ac_code_num+1)/Decimal(2020)

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small_ac)*r
            max_bound = min_bound + Decimal(big_ac-small_ac)*r

            #manuall 4 characters flight details
            for fd in flight_details:
                small, big = self.domain2_flight_boundaries[fd]
                #print(small)
                #print(big)

                r = Decimal(max_bound-min_bound)

                min_bound += Decimal(small)*r
                max_bound = min_bound + Decimal(big-small)*r
                

            #date
            small_date = Decimal(day_diff)/Decimal(1008)
            big_date = Decimal(day_diff+1)/Decimal(1008)

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small_date)*r
            max_bound = min_bound + Decimal(big_date-small_date)*r

            #truncation
            str_min = str(min_bound)
            str_max = str(max_bound)
            val = ""
            for i in range(len(str_max)):
                val += str_max[i]
                if str_max[i] != str_min[i]:
                    break

            return str(val)[2:]

        elif domain_number == 3:

            message_list = []

            msg = message[1:]

            def find_list(password, message_list):
                print(password)
                print(message_list)

                if len(password) == 0:
                    return message_list

                if password[0].isdigit():
                    temp = password[1:]
                    return find_list(temp, message_list.copy()+[password[0]])

                else:
                    for sub_m in self.domain3_boundaries.keys():
                        if password.find(sub_m) == 0:
                            temp = password[len(sub_m):]

                            ret = find_list(temp, message_list.copy()+[sub_m])
                            if ret != False:
                                return ret
                    
                    return False
            
            mg_list = find_list(msg, message_list)
            val = None
            for i in range(len(mg_list),-1,-1):
                val = Decimal(self.get_arithmatic_code(mg_list[:i], self.domain3_boundaries, stop_symbol=None))
                if int(str(val)[2:]) < math.factorial(52):
                    break
            
            return str(val)[2:]

        elif domain_number == 4:
            message_split = message_list = message.split(",")
            left = message_split[0].strip().split(" ")

            coord_1 = left[0]
            dir_1 = left[1]

            right = message_split[1].strip().split(" ")

            coord_2 = right[0]
            dir_2 = right[1]

            #range of latitude

            latitude_val = int(coord_1.replace(".",""))
            #print(latitude_val)

            #range of longitude
            longitude_val = int(coord_2.replace(".",""))
            #print(longitude_val)

            #get arithmatic code manually
            min_bound = Decimal(0)
            max_bound = Decimal(1)

            small_lat = latitude_val/900001
            big_lat = (latitude_val+1)/900001

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small_lat)*r
            max_bound = min_bound + Decimal(big_lat-small_lat)*r

            #get arithmatic code manually
            small_long = longitude_val/1800001
            big_long = (longitude_val+1)/1800001

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small_long)*r
            max_bound = min_bound + Decimal(big_long-small_long)*r

            #truncation
            str_min = str(min_bound)
            str_max = str(max_bound)
            val = ""
            for i in range(len(str_max)):
                val += str_max[i]
                if str_max[i] != str_min[i]:
                    break
    
            directions = {"NE":1, "NW":2, "SE":3, "SW":4}
            direction_key = str(dir_1)+str(dir_2)

            return str(directions[direction_key])+str(val)[2:]

        elif domain_number == 5:
            #print(self.domain5_stname_to_num)
            #print(self.domain5_stsuffix_to_num)
            
            number = message[0:message.find(" ")]
            number = "0"*(4-len(str(number))) + str(number)
            #print(number)

            rest = message[message.find(" ")+1:].strip()

            st_name = None
            st_suffix = None
            for stname in self.domain5_stname_to_num.keys():

                if st_name is not None and st_suffix is not None:
                    break

                if rest.find(stname) == 0:
                    #print(stname)
                    for suffix in self.domain5_stsuffix_to_num:
                        if rest == stname + " " + suffix:
                            st_name = self.domain5_stname_to_num[stname]
                            st_suffix = self.domain5_stsuffix_to_num[suffix]
                            break

            
            #artithmetic encoding
            min_bound = Decimal(0)
            max_bound = Decimal(1)

            small_num = Decimal(number)/Decimal(9999+1)
            big_num = Decimal(int(number)+1)/Decimal(9999+1)

            r = Decimal(max_bound-min_bound)
            #print(r)

            min_bound += Decimal(small_num)*r
            max_bound = min_bound + Decimal(big_num-small_num)*r

            #print(min_bound)
            #print(max_bound)

            #get arithmatic code manually
            small_name = Decimal(st_name)/Decimal(len(self.domain5_stname_to_num)+1)
            big_name = Decimal(st_name+1)/Decimal(len(self.domain5_stname_to_num)+1)

            r = Decimal(max_bound-min_bound)

            min_bound += Decimal(small_name)*r
            max_bound = min_bound + Decimal(big_name-small_name)*r
            #print(min_bound)
            #print(max_bound)

            #get arithmatic code manually
            small_suff= st_suffix/(len(self.domain5_stsuffix_to_num)+1)
            big_suff = (st_suffix+1)/(len(self.domain5_stsuffix_to_num)+1)

            r = Decimal(max_bound-min_bound)
            #print(r)

            min_bound += Decimal(small_suff)*r
            max_bound = min_bound + Decimal(big_suff-small_suff)*r
            #print(min_bound)
            #print(max_bound)

            #truncation
            str_min = str(min_bound)
            str_max = str(max_bound)
            val = ""
            for i in range(len(str_max)):
                val += str_max[i]
                if str_max[i] != str_min[i]:
                    break
            
            return str(val)[2:]

        elif domain_number == 6:
            
            message_list = message.split(" ")

            val = Decimal(self.get_arithmatic_code(message_list, self.domain6_boundaries, stop_symbol=None))
            return str(val)[2:]

        elif domain_number == 7:
            message_list = message.split(" ")

            if len(message_list[-1] ) == 0:
                message_list.pop()

            print(message_list)

            val = Decimal(self.get_arithmatic_code(message_list, self.domain7_boundaries, stop_symbol=None))
            return str(val)[2:]

        elif domain_number == 8:

            message_list = message.split(" ")
            #print(message_list)

            val = Decimal(self.get_arithmatic_code(message_list, self.domain8_boundaries, stop_symbol=None))
            #print("val: ", len(self.domain8_boundaries))
            return str(val)[2:]
        
        else:
        
            return None
            
        

    def encode(self, message):
        print(message)

        domain = identify_domain(message)
        print(domain)

        message_length=len(message)

        val = self.encode_helper(message, domain)

        if val is None:
            print(message + " : " + str(domain))
            return None
        else:
            #print("Val:", val)
            pass
        
        #"0"*(2-len(str(message_length)))
        final_val = int(str(domain)+ val + "0"*(2-len(str(message_length)))+str(message_length))

        #encode to a card sequence
        deck = list(range(0,52))
        
        #padded_cards = list(range(0,PADDING_CARDS))
        #arith_cards = list(range(PADDING_CARDS, 52))
        print("final val: ", final_val)
        encoded_deck = number_to_cards(final_val, deck)

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
        #print("decode_val:",val)

        domain = int(str(val)[0])

        message_length = int(str(val)[-2:])

        message = str(val)[1:-2]
        #print(message)
    
        if message_length is None:
            return None, "NULL"

        if domain == 8:
            #print("0."+message)
            return message_length, self.get_word(Decimal("0."+message), self.domain8_boundaries, message_length, delimiter=" ")

        elif domain == 7:
            #print("0."+message)
            return message_length, self.get_word(Decimal("0."+message), self.domain7_boundaries, message_length, delimiter=" ")

        elif domain == 6:
            #print("0."+message)
            return message_length, self.get_word(Decimal("0."+message), self.domain6_boundaries, message_length, delimiter=" ")

        elif domain == 5:
            return message_length, self.get_word_domain5(Decimal("0."+message), message_length, delimiter=" ")

        elif domain == 4:
            if len(message) <= 2:
                return None, "NULL"

            directions = {1:"NE", 2:"NW", 3:"SE", 4:"SW"}

            if int(message[0]) > 4 or int(message[0]) == 0:
                return None, "NULL"

            direct = directions[int(message[0])]

            return message_length, self.get_word_domain4(Decimal("0."+message[1:]), message_length, direct, delimiter=" ")

        elif domain == 3:
            #print("0."+message)
            return message_length, self.get_word(Decimal("0."+message), self.domain3_boundaries, message_length, delimiter="", domain_3 = True)

        elif domain == 2:
            return message_length, self.get_word_domain2(Decimal("0."+message), message_length)

        elif domain == 1:
            return message_length, self.get_word(Decimal("0."+message), self.domain1_boundaries, message_length, delimiter="")

        else:
            return None, "NULL"


    def decode(self, deck):
        #print("Decoding")

        word_count = {}
        max_word_count = 0
        max_word = None
        message_length=0

        #try all encoding lengths
        for i in range(3, 52):
            #print(i)
            word_length, word = self.decode_helper(i, deck)
            #print(word)
            #print(len(word))

            #word cannot be Null
            if word != "NULL":
                word_count[word] = word_count[word] + 1 if word in word_count else 1

                #Choose the best word (should appear at least once times)
                if word_count[word] > max_word_count:
                    max_word_count = word_count[word]
                    max_word = word
                    message_length = word_length

                if max_word_count >= 5:
                    break
                
        #print(word_count)
        if max_word in word_count and word_count[max_word] > 1 and message_length>0:
            if message_length == len(max_word):
                return max_word
            else:
                return "PARTIAL: "+max_word[:message_length]
        else:
            return "NULL"

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
    message = "5sot xdgbv"
    agent = ArtihmaticCodingAgent()
    print("Message Length:", len(message))
    deck = agent.encode(message)
    print(agent.decode(deck))
