import argparse
import os
import random
import sys
import numpy as np

DICT = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
DIGITS = [str(i) for i in range(10)]
K_MAX = 5 # max number of words and digit randomly selected each
K_MIN = 2
WORD_PROBS = [0.4, 0.3, 0.2, 0.1]
NUM_PROBS = [0.25, 0.25, 0.25, 0.25]

def load_dictionary(filepath=None):
	"""Loads a dictionary into the global @DICT.
	
	Expects the file @ filepath to be newline delimited list of words.
	"""
	global DICT

	if filepath is not None:
		try:
			with open(filepath, 'r') as f:
				lines = [line.rstrip() for line in f]
				# print(lines[:10])
				DICT = lines
		except OSError:
			print(f"Load dictionary failed: could not open/read file: {filepath}")
			sys.exit()

def get_random_num_from_distribution(min, max, probs):
	return np.random.choice(np.arange(min, max+1), p=probs)

def gen_msgs(n=1, seed=1):
	"""Generates a formatted message containing username and password.

	Message format: @ followed by a sequence of words and digits in random
	order. There are @K words and @K digits.

	The first word in the message is the username and the rest is the password.
	"""
	rng = random.Random(seed)
	msgs = []

	for _ in range(n):
		k_digits = get_random_num_from_distribution(K_MIN, K_MAX, NUM_PROBS)
		k_words = get_random_num_from_distribution(K_MIN, K_MAX, WORD_PROBS)
		selected_digits = [rng.choice(DIGITS) for _ in range(k_digits)]
		selected_words = rng.choices(DICT, k=k_words)
		merged_selection = selected_digits + selected_words

		rng.shuffle(merged_selection)

		msgs.append(f"@{''.join(merged_selection)}")

	return msgs

def write_msgs(out_file, msgs):
	"""Writes a list of string messages to a file delimited by newline."""
	with open(out_file, "w") as f:
		f.write("\n".join(msgs))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        default = "",
        help = "Required: filepath of the dictionary file to select words from."
    )
	parser.add_argument(
        "--output_file",
        "-O",
        type=str,
        default = "password_messages.txt",
        help = "number of password messages to generate. Default is password_messages.txt"
    )
	parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default = 42,
        help = "seed used by random number generator. Default is 42. Specify 0 to generate random seed."
    )

	parser.add_argument(
        "--num",
        "-n",
        type=int,
        default = 100,
        help = "number of password messages to generate. Default is 100."
    )

	if len(sys.argv)==1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	args = parser.parse_args()
	assert os.path.exists(args.filepath), f"dictionary file not found: {args.filepath}"

	filepath = args.filepath
	num_messages = args.num
	seed = args.seed
	out_file = args.output_file

	load_dictionary(filepath)
	write_msgs(out_file, gen_msgs(n=num_messages, seed=seed))