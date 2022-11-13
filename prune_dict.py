from email.headerregistry import Address
from enum import Enum
import random
import string
import math
from collections import defaultdict
import os
import enchant
import hashlib
import numpy as np

UNTOK = '*'
EMPTY = ''
DICT_SIZE = 27000
SENTENCE_LEN = 6
ENGLISH_DICTIONARY = enchant.Dict("en_US") # pip install pyenchant

with open("./messages/agent7/words.txt", "w") as o:
    with open("./messages/agent7/30k.txt", "r") as f:
        line = f.readline()
        while line:
            line = line.strip()
            if ENGLISH_DICTIONARY.check(line):
                o.write(line + '\n')
            line = f.readline()

