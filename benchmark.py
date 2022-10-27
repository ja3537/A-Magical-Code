import argparse
import multiprocessing
import random
from importlib import import_module
from itertools import product
from string import ascii_lowercase

import matplotlib.pyplot as plt
import numpy as np

# ==============
# Argparse setup
# ==============
parser = argparse.ArgumentParser()
parser.add_argument(
    "--agent",
    "-a",
    type=int,
    required=True,
    help="which agent to use from 1 to 8",
)
parser.add_argument(
    "--trials",
    "-t",
    default=50,
    type=int,
    help="number of trials per length/shuffle datapoint. defaults to 50",
)
parser.add_argument(
    "--disable-threading",
    "-d",
    default=False,
    action="store_true",
    help="disable multithreading for benchmarking. makes script run on systems where multithreading isn't working but makes benchmark run more slowly",
)
pargs = parser.parse_args()

agent_name = f"agent{pargs.agent}"
agent_module = import_module(f".{agent_name}", "agents")


# ==================
# Message Generators
# ==================
def get_uniform_message(length: int) -> str:
    return "".join(random.choices(ascii_lowercase, k=length))


# ===========================
# Benchmarking infrastructure
# ===========================
def shuffle(n, deck):
    rng = np.random.default_rng()
    shuffles = rng.integers(0, 52, n)
    for pos in shuffles:
        top_card = deck[0]
        deck = deck[1:]
        deck = deck[:pos] + [top_card] + deck[pos:]
    return deck


def run_trial(args: tuple[int, int]) -> tuple[int, int, float]:
    agent = agent_module.Agent()
    length, n = args
    successes = 0
    for _ in range(pargs.trials):
        word = get_uniform_message(length)
        deck = agent.encode(word)
        shuffled = shuffle(n, deck)
        out = agent.decode(shuffled)

        if word == out:
            successes += 1
    return length, n, successes / pargs.trials


lengths = list(range(1, 11))
shuffle_amounts = list(range(100))
work = list(product(lengths, shuffle_amounts))
pool = multiprocessing.Pool(multiprocessing.cpu_count())
if pargs.disable_threading:
    results = map(run_trial, work)
else:
    results = pool.imap_unordered(run_trial, work, chunksize=4)

print("Collecting results...")
collected_results = {}
for result in results:
    length, n, recovery_rate = result
    if not length in collected_results:
        collected_results[length] = []
    collected_results[length].append((n, recovery_rate))

# ======================
# Plot benchmark results
# ======================
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
