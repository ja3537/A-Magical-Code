import numpy as np
import cards
from agents.default import Agent as default_agent

class Mission:
    def __init__(self, args):
        self.seed = args.seed
        self.verbose = args.verbose
        self.rng = np.random.default_rng(self.seed)
        #self.agent = args.agent[0]
        self.agent = default_agent()
        self.n = int(args.n)
        self.null_rate = float(args.null_rate)
        self.output = args.output
        self.messages = []
        with open(args.messages, 'r') as f:
            text_lines = f.read().splitlines()
            i = 0
            while i < len(text_lines):
                if(text_lines[i] == ''):
                    i = i + 1
                    continue
                null_roll = self.rng.random()
                if null_roll > (1-self.null_rate):
                    self.messages.append("NULL")
                else:
                    self.messages.append(text_lines[i])
                    i = i + 1

        self.encoded_decks = []
        self.shuffled_decks = []
        self.decoded = [None]*len(self.messages)
        self.scores = [None]*len(self.messages)
        self.total_score = 0
        self.messages_index = list(range(len(self.messages))) #indexes of messages not yet used in mission
        self.rng.shuffle(self.messages_index) #determines random order for messages during decoding

    def execute_mission(self):
        for m in self.messages: #encode stage
            if m != "NULL":
                d = self.agent.encode(m)
                if(not cards.valid_deck(d)):
                    print("invalid deck")
            else:
                d = cards.generate_deck(self.rng, random = True)
            self.encoded_decks.append(d)

        for i in range(len(self.encoded_decks)): #shuffling stage
            d = self.encoded_decks[i]
            if cards.valid_deck(d):
                self.shuffled_decks.append(self.s(self.n, d))



        for i in self.messages_index:
            s_deck = self.shuffled_decks[i]
            if cards.valid_deck(s_deck):
                decoded_m = self.agent.decode(s_deck)
                score = self.score_message(self.messages[i], decoded_m)
                self.decoded[i] = decoded_m
            else:
                score = 0
                self.decoded[i] = "invalid deck: {}".format(e_deck)
            self.scores[i] = score
            self.total_score += score


        self.make_output_file()




    def score_message(self, m, decoded_m):
        if m == decoded_m:
            return 1
        else:
            return 0

    def s(self, n, deck):
        shuffles = self.rng.integers(0, 52, n)
        for pos in shuffles:
            top_card = deck[0]
            deck = deck[1:]
            deck = deck[:pos] + [top_card] + deck[pos:]
        return deck

    def make_output_file(self):
        with open(self.output, 'w+') as f:
            for i in range(len(self.messages)):
                f.write(self.messages[i] + '\n')
                if(self.verbose):
                    f.write('encoded deck: ' + str(self.encoded_decks[i]) + '\n')
                    f.write('shuffled deck: ' + str(self.shuffled_decks[i]) + '\n')
                f.write(self.decoded[i] + '\n')
                f.write(str(self.scores[i]) + '\n')
                f.write('\n')
            f.write('total score for agent: {}/{}'.format(self.total_score, len(self.messages)))






