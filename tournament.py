import argparse
from mission import Mission
import time

#must specify batch number and agent number. Script will iterate through all messages from each domain of that batch number for that player
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default = 42,
        help = "seed used by random number generator. Default is 42. Specify 0 to generate random seed."
    )
    parser.add_argument(
        "--agent",
        "-a",
        default = 'd',
        nargs = 1,
        help = "which agent to use from 1 to 8. d for default"
    )
    parser.add_argument(
        "--messages",
        "-m",
        default = "messages/default/example.txt",
        help = "name of file holding messages to encode"
    )
    parser.add_argument(
        "--output",
        "-o",
        default = "output.txt",
        help = "name of the output file where decoded messages and scores are stored"
    )
    parser.add_argument(
        "-n",
        default = 10,
        help = "set number of shuffles per deck for this mission"
    )
    parser.add_argument(
        "--null_rate",
        "-nr",
        default = 0,
        help = "set the probablility (between 0 and 1) of giving agent a random deck instead of a message"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default = False,
        action="store_true",
        help = "verbose mode includes encrypted and shuffled decks in output"

    )
    parser.add_argument(
        "--rand_n",
        "-rn",
        default = False,
        action = "store_true",
        help = "(boolean) sets n to be a random value up to N for each deck"
    )
    parser.add_argument(
        "--runs",
        "-r",
        default = 1,
        type = int,
        help = "number of runs per message in message file"
    )
    parser.add_argument(
        "--csv",
        "-csv",
        default=False,
        action="store_true",
        help="(boolean) outputs a csv file instead of txt file"
    )
    parser.add_argument(
        "--batch",
        "-b",
        default = 1,
        help="batch number, out of 4, specified for the tournament"
    )


args = parser.parse_args()
assert float(args.null_rate) >= 0 and float(args.null_rate) < 1, "null rate must be between 0 (inclusive) and 1 (exclusive)"
assert args.runs >= 1, "number of runs must be a positive integer"

n_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
args.csv = True

total_start = time.time()
for domain in range(1,9):
    start = time.time()
    seed_file = "tournament_seeds/seeds_{}_{}.txt".format(domain, args.batch)
    with open(seed_file, 'r') as f:
        seeds = f.read().splitlines()

    for i in range(len(n_values)):
        tick = time.time()
        args.seed = int(seeds[i])
        args.n = n_values[i]
        messages_file = 'messages/tournament/m_{}_{}.txt'.format(domain, args.batch)
        args.messages = messages_file
        output_file = "tournament_results/results_{}_{}_{}_{}.csv".format(args.agent[0], domain, args.batch, args.n)
        args.output = output_file

        mission = Mission(args)

        mission.execute_mission()
        tock = time.time()
        print("mission completed for agent {} in domain {} and batch {} for n value of {} in {:.2f}s".format(args.agent[0], domain, args.batch, args.n, tock-tick))
    end = time.time()

    print("mission completed for agent {} in domain {} and batch {} in a total time of {:.2f}s".format(args.agent[0], domain, args.batch, end-start))

total_end = time.time()
print("all domains completed for agent {} with batch {} in a total time of {:.2f}s".format(args.agent[0], args.batch, total_end - total_start))



#mission = Mission(args)
#mission.execute_mission()