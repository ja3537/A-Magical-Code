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
        default = "messages.txt",
        help = "name of file holding messages to encode"
    )
    parser.add_argument(
        "--output",
        "-o",
        default = "output.txt",
        help = "name of the output file where decoded messages and scores are stored"
    )
    #TODO: timeout, null interleaving rate, printing results


args = parser.parse_args()
print(args)

mission = Mission(args)
mission.execute_mission()