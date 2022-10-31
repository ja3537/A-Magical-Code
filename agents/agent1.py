import random
from itertools import permutations

import numpy as np

import cards


class Agent:
    def __init__(self):
        self.encode_len = 7  # Total items = nPn
        self.valid_cards = list(range(52-self.encode_len, 52))
        self.all_encodings = list(permutations(self.valid_cards))

        # Out of all the encodings generated, keep only a percentage
        # This gives us some guard against NULL decks
        self.percentage_valid = 0.2
        num_valid = int(0.2 * len(self.all_encodings))
        self.valid_encodings = self.all_encodings[:num_valid]

        self.dict_encode = {str(idx): self.valid_encodings[idx] for idx in range(len(self.valid_encodings))}
        self.dict_decode = {self.valid_encodings[idx]: str(idx) for idx in range(len(self.valid_encodings))}
        self.seed = 0
        self.rng = np.random.default_rng(self.seed)

    def encode(self, message):
        if message not in self.dict_encode:
            return ValueError(f"message is not valid. Must be an int less than: {len(self.valid_encodings)}")

        seq_encode = list(self.dict_encode[message])
        # Scramble the rest of the cards above message
        seq_rand = list(range(0, 52-self.encode_len))
        self.rng.shuffle(seq_rand)
        seq_total = seq_rand + seq_encode
        return seq_total

    def decode(self, deck):
        encoding = [c for c in deck if c in self.valid_cards]
        encoding = tuple(encoding)
        if encoding in self.valid_encodings:
            return self.dict_decode[encoding]
        else:
            return "NULL"

#---------------------------huffman stuff! ---------------------------
# A Huffman Tree Node
import heapq
 
class node:
    def __init__(self, freq, symbol, left=None, right=None):
        # frequency of symbol
        self.freq = freq
 
        # symbol name (character)
        self.symbol = symbol
 
        # node left of current node
        self.left = left
 
        # node right of current node
        self.right = right
 
        # tree direction (0/1)
        self.huff = ''
        self.encoding_dict = {}
         
    def __lt__(self, nxt):
        return self.freq < nxt.freq
         
 
    # utility function to print huffman
    # codes for all symbols in the newly
    # created Huffman tree
    def printNodes(self, node, val=''):
        
        # huffman code for current node
        newVal = val + str(node.huff)
    
        # if node is not an edge node
        # then traverse inside it
        if(node.left):
            self.printNodes(node.left, newVal)
            
        if(node.right):
            self.printNodes(node.right, newVal)
    
            # if node is edge node then
            # display its huffman code
        if(not node.left and not node.right):
            self.encoding_dict[node.symbol]= newVal
 
 
# characters for huffman tree
chars = ['e', 'm', 'a', 'h', 'r', 'g', 'i', 'b', 'o', 'f', 't', 'y', 'n', 'w', 's', 
'k', 'l', 'v', 'c', 'x', 'u', 'z', 'd', 'j', 'p', 'q']
 
# frequency of characters
freq = [11.1607, 3.0129, 8.4966, 3.0034, 7.5809, 2.4705, 7.5448, 2.072, 7.1635, 1.8121, 6.9509, 1.7779, 6.6544, 1.2899, 5.7351, 1.1016, 5.4893, 1.0074, 4.5388, 0.2902,
 3.6308, 0.2722, 3.3844, 0.1965, 3.1671, 0.1962]
 
# list containing unused nodes
nodes = []
 
# converting characters and frequencies
# into huffman tree nodes
for x in range(len(chars)):
    heapq.heappush(nodes, node(freq[x], chars[x]))
 
while len(nodes) > 1:
     
    # sort all the nodes in ascending order
    # based on their frequency
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)
 
    # assign directional value to these nodes
    left.huff = 0
    right.huff = 1
 
    # combine the 2 smallest nodes to create
    # new node as their parent
    newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)
 
    heapq.heappush(nodes, newNode)
 
# Huffman Tree is ready!
s = nodes[0].printNodes(nodes[0])

if __name__ == "__main__":
    agent = Agent()
    deck = agent.encode("1")
    valid_deck = cards.valid_deck(deck)
    msg = agent.decode(deck)

    # Check NULL redundancy
    count_null = 0
    for idx in range(1000):
        rng = np.random.default_rng(idx)
        deck_r = cards.generate_deck(rng, random=True)
        msg_r = agent.decode(deck_r)
        if msg_r == "NULL":
            count_null += 1
    print("Done")
##---------------------------huffman_decoding---------------------------##
#Huffman_Decoding('11111110',nodes[0])
def Huffman_Decoding(encoded_data, huffman_tree):
    tree_head = huffman_tree
    decoded_output = []
    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right   
        elif x == '0':
            huffman_tree = huffman_tree.left
        try:
            if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                pass
        except AttributeError:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head
        
    string = ''.join([str(item) for item in decoded_output])
    return string