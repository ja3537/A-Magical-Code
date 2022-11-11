import argparse
from mission import Mission


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


args = parser.parse_args()
assert float(args.null_rate) >= 0 and float(args.null_rate) < 1, "null rate must be between 0 (inclusive) and 1 (exclusive)"
assert args.runs >= 1, "number of runs must be a positive integer"

mission = Mission(args)
mission.execute_mission()