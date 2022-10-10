import numpy as np
import cards
from agents.default import Agent as default_agent

class Mission:
    def __init__(self, args):
        self.seed = args.seed
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
                null_roll = self.rng.random()
                if null_roll > (1-self.null_rate):
                    self.messages.append("NULL")
                else:
                    self.messages.append(text_lines[i])
                    i = i + 1

        self.encoded_decks = []
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

        for d in self.encoded_decks: #shuffling stage
            d = self.s(self.n, d)


        for i in self.messages_index:
            decoded_m = self.agent.decode(self.encoded_decks[i])
            score = self.score_message(self.messages[i], decoded_m)
            self.scores[i] = score
            self.total_score = self.total_score + score
            self.decoded[i] = decoded_m

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
                f.write(self.decoded[i] + '\n')
                f.write(str(self.scores[i]) + '\n')
                f.write('\n')
            f.write('total score for agent: {}/{}'.format(self.total_score, len(self.messages)))






