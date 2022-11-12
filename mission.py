import numpy as np
import cards
import constants
from importlib import import_module
from agents.default import Agent as default_agent
import csv

class Mission:
    def __init__(self, args):
        self.seed = args.seed
        self.verbose = args.verbose
        self.runs = args.runs
        self.csv = args.csv
        self.agent_number = args.agent[0]
        if(self.seed != 0):
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()
        if args.agent[0] in constants.possible_agents:
            if args.agent[0] == 'd':
                self.agent = default_agent()
            else:
                agent_name = "agent{}".format(args.agent[0])
                agent_module = import_module(f".{agent_name}", "agents")
                self.agent = agent_module.Agent()
        else:
            print("error loading agent ", args.agent[0])
            exit()
        self.n = int(args.n)
        self.rand_n = args.rand_n
        self.null_rate = float(args.null_rate)
        self.output = args.output
        self.messages = []
        with open(args.messages, 'r') as f:
            text_lines = f.read().splitlines()
            i = 0
            while i < len(text_lines):
                if(text_lines[i] == ''):
                    i += 1
                    continue
                null_roll = self.rng.random()
                if null_roll > (1-self.null_rate):
                    self.messages.append("NULL")
                else:
                    self.messages.append(text_lines[i])
                    i += 1

        self.encoded_decks = []
        self.shuffled_decks = []
        self.n_values = []
        self.decoded = []
        self.scores = []
        for i in range(len(self.messages)):
            self.decoded.append([None]*self.runs)
            self.scores.append([None] * self.runs)
        self.total_score = 0
        self.messages_index = list(range(len(self.messages)*self.runs)) #indexes of messages not yet used in mission
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
            for run in range(self.runs):
                d = self.encoded_decks[i]
                if cards.valid_deck(d):
                    if(self.rand_n):
                        n = self.rng.integers(1, self.n + 1, 1)[0]
                    else:
                        n = self.n
                    if run == 0:
                        self.n_values.append([n])
                        self.shuffled_decks.append([self.s(n, d)])
                    else:
                        self.n_values[i].append(n)
                        self.shuffled_decks[i].append(self.s(n, d))
                else:
                    if run == 0:
                        self.n_values.append([n])
                        self.shuffled_decks.append([d])
                    else:
                        self.n_values[i].append(n)
                        self.shuffled_decks[i].append(d)



        for i in self.messages_index:
            m_index = i % len(self.messages)
            run = i // len(self.messages)
            s_deck = self.shuffled_decks[m_index][run]
            if cards.valid_deck(s_deck):
                decoded_m = self.agent.decode(s_deck)
                score = self.score_message(self.messages[m_index], decoded_m)
                self.decoded[m_index][run] = decoded_m
            else:
                score = 0
                self.decoded[m_index][run] = "invalid deck: {}".format(s_deck)
            self.scores[m_index][run] = score
            self.total_score += score


        self.make_output_file()


    def score_partial(self, m, decoded_m):
        if m.startswith(decoded_m):
            return len(decoded_m)/(len(m) + 1)
        else:
            return 0

    def score_message(self, m, decoded_m):
        if decoded_m.startswith("PARTIAL: "):
            stripped_m = decoded_m.removeprefix("PARTIAL: ")
            return self.score_partial(m, stripped_m)
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


    def make_output_csv_file(self):
        if self.output == 'output.txt':
            self.output = 'output.csv'

        with open(self.output, 'w+', newline='') as f:
            header = ['agent', 'seed', 'n', 'message', 'decoded message', 'score']
            writer = csv.writer(f)
            writer.writerow(header)

            for i in range(len(self.messages)):
                row = [str(self.agent_number), str(self.seed), str(self.n), self.messages[i], self.decoded[i][0], self.scores[i][0]]
                writer.writerow(row)


    def make_output_text_file(self):
        with open(self.output, 'w+') as f:
            for i in range(len(self.messages)):
                f.write(self.messages[i] + '\n')
                for run in range(self.runs):
                    if(self.verbose):
                        if run == 0:
                            f.write('encoded deck: ' + str(self.encoded_decks[i]) + '\n')
                        f.write('n value: ' + str(self.n_values[i][run]) + '\n')
                        f.write('shuffled deck: ' + str(self.shuffled_decks[i][run]) + '\n')
                    f.write(self.decoded[i][run] + '\n')
                    f.write("score " + str(self.scores[i][run]) + '\n')
                f.write("total score for agent on this message: {:.3f}/{}\n".format(sum(self.scores[i]), self.runs))
                f.write('\n')
            f.write('\n')
            f.write('total score for agent: {:.3f}/{}'.format(self.total_score, len(self.messages)*self.runs))


    def make_output_file(self):
        if self.csv:
            self.make_output_csv_file()
        else:
            self.make_output_text_file()



