import argparse
import numpy as np
import random

corpus_lengths = {
    '1':31057,
    '2':47013,
    '3':44435,
    '4':41834,
    '5':39254,
    '6':36712,
    '7':34238,
    '8':31830,
    '9':29497,
    'corpus':2564
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default = 42,
        help = "seed used by random number generator. Default is 42. Specify 0 to generate random seed."
    )

    parser.add_argument(
        "--ngram",
        "-g",
        type=int,
        default = None,
        help = "Specifies length of ngram to return. If none will return a full sentence from corpus. Default is None."
    )

    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default = 5,
        help = "Number of tests to generate. Default is 5."
    )

    parser.add_argument(
        "--output",
        "-o",
        default = 'test.txt',
        help = "Name of output file. Default is 'test.txt'."
    )

    args = parser.parse_args()

    if(args.seed != 0):
            rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    random.seed(rng)

    if (args.ngram):
        if args.ngram > 9:
            ng = 9
        else:
            ng = args.ngram

        file_name = "corpus-ngram-"+str(ng)+".txt"
        indexes = random.sample(range(0, corpus_lengths[str(ng)]), args.num)
        returnable = []
        with open(file_name) as fp:
            for i, line in enumerate(fp):
                if i in indexes:
                    returnable.append(line[:len(line)-1])

    else:
        indexes = random.sample(range(0, 2564), args.num)
        returnable = []
        with open("unedited_corpus.txt") as fp:
            for i, line in enumerate(fp):
                if i in indexes:
                    returnable.append(line[:len(line)-1])

    with open(args.output, 'w') as f:
        for i,line in enumerate(returnable):
            if i != len(returnable)-1:
                f.write(f"{line}\n")
            else:
                f.write(f"{line}")


