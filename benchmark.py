import argparse
import multiprocessing
import random
import sys
from functools import partial
from importlib import import_module
from itertools import product
from string import ascii_letters, ascii_lowercase, digits, punctuation

import matplotlib.pyplot as plt
import numpy as np


# ==================
# Message Generators
# ==================
def uniform_random_message(character_set: str, length: int) -> str:
    return "".join(random.choices(character_set, k=length))


# ===============
# Message Domains
# ===============
# name -> generator(length)
MESSAGE_DOMAINS = {
    "lower": partial(uniform_random_message, ascii_lowercase),
    "alpha": partial(uniform_random_message, ascii_letters),
    "alphanum": partial(uniform_random_message, ascii_letters + digits),
    "password": partial(uniform_random_message, ascii_letters + digits + punctuation),
}

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
    "--min-length",
    "-L",
    default=1,
    type=int,
    help="minimum message length. defaults to 1",
)
parser.add_argument(
    "--max-length",
    "-l",
    default=10,
    type=int,
    help="maximum message length. defaults to 10",
)
parser.add_argument(
    "--min-shuffles",
    "-N",
    default=0,
    type=int,
    help="minimum shuffle amount. defaults to 0",
)
parser.add_argument(
    "--max-shuffles",
    "-n",
    default=100,
    type=int,
    help="maximum shuffle amount. defaults to 100",
)
parser.add_argument(
    "--domain",
    "-d",
    default="lower",
    help="message domain. one of: " + ", ".join(MESSAGE_DOMAINS.keys()),
)
parser.add_argument(
    "--disable-threading",
    default=False,
    action="store_true",
    help="disable multithreading for benchmarking. makes script run on systems where multithreading isn't working but makes benchmark run more slowly",
)
pargs = parser.parse_args()

agent_name = f"agent{pargs.agent}"
agent_module = import_module(f".{agent_name}", "agents")


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


# Suppress stdout
# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
class DummyFile(object):
    def write(self, x):
        pass

    def flush(self):
        pass


def run_trial(args: tuple[int, int]) -> tuple[int, int, float]:
    # Suppress stdout
    save_stdout = sys.stdout
    sys.stdout = DummyFile()

    agent = agent_module.Agent()
    length, n = args
    successes = 0
    for _ in range(pargs.trials):
        word = MESSAGE_DOMAINS[pargs.domain](length)
        deck = agent.encode(word)
        shuffled = shuffle(n, deck)
        out = agent.decode(shuffled)

        if word == out:
            successes += 1

    # Restore stdout
    sys.stdout = save_stdout

    return length, n, successes / pargs.trials


lengths = list(range(pargs.min_length, pargs.max_length + 1))
shuffle_amounts = list(range(pargs.min_shuffles, pargs.max_shuffles + 1))
work = list(product(lengths, shuffle_amounts))
pool = multiprocessing.Pool(multiprocessing.cpu_count())

print("Collecting results...")

if pargs.disable_threading:
    results = map(run_trial, work)
else:
    results = pool.imap_unordered(run_trial, work, chunksize=4)

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
plt.title(f"Shuffle Count vs. Recovery Rate (Agent {pargs.agent})")
plt.xlabel("Number of shuffles")
plt.ylabel("Recovery rate")
plt.savefig("benchmark.png", dpi=300)
# print(
#     f"{round((agent.decoding_errors / agent.total_decode) * 100, 2)}% ({agent.decoding_errors}/{agent.total_decode}) decoding errors"
# )
