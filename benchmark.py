import multiprocessing
import random
from itertools import product
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np

from agents.agent8 import Agent

TRIALS = 50


def get_uniform_message(length: int) -> str:
    return "".join(random.choices(ascii_lowercase, k=length))


def shuffle(n, deck):
    rng = np.random.default_rng()
    shuffles = rng.integers(0, 52, n)
    for pos in shuffles:
        top_card = deck[0]
        deck = deck[1:]
        deck = deck[:pos] + [top_card] + deck[pos:]
    return deck


def run_trial(args: tuple[int, int]) -> tuple[int, int, float]:
    agent = Agent()
    length, n = args
    successes = 0
    for _ in range(TRIALS):
        word = get_uniform_message(length)
        deck = agent.encode(word)
        shuffled = shuffle(n, deck)
        out = agent.decode(shuffled)

        if word == out:
            successes += 1
    return length, n, successes / TRIALS


lengths = list(range(1, 11))
shuffle_amounts = list(range(100))
work = list(product(lengths, shuffle_amounts))
pool = multiprocessing.Pool(multiprocessing.cpu_count())
results = pool.imap_unordered(run_trial, work, chunksize=4)

print("Collecting results...")
collected_results = {}
for result in results:
    length, n, recovery_rate = result
    if not length in collected_results:
        collected_results[length] = []
    collected_results[length].append((n, recovery_rate))

print("Plotting...")
for length in collected_results:
    ys = [recovery_rate for _, recovery_rate in sorted(collected_results[length])]
    plt.plot(shuffle_amounts, ys, label=str(length))

plt.legend()
plt.title("Shuffle Count vs. Recovery Rate Across Message Lengths")
plt.xlabel("Number of shuffles")
plt.ylabel("Recovery rate")
plt.savefig("benchmark.png", dpi=300)
# print(
#     f"{round((agent.decoding_errors / agent.total_decode) * 100, 2)}% ({agent.decoding_errors}/{agent.total_decode}) decoding errors"
# )
