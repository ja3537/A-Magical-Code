import random
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


agent = Agent()

for length in range(1, 11):
    xs = list(range(100))
    ys = []
    for n in xs:
        successes = 0
        for _ in range(TRIALS):
            word = get_uniform_message(length)
            deck = agent.encode(word)
            shuffled = shuffle(n, deck)
            print(f"Shuffled {n} times:", shuffled)
            out = agent.decode(shuffled)

            if word == out:
                successes += 1
                print("success")
        ys.append(successes / TRIALS)
    plt.plot(xs, ys, label=str(length))
plt.legend()
plt.savefig("benchmark.png", dpi=300)
plt.title("Exact match recovery rate for uniform random messages of different lengths")
plt.xlabel("Number of shuffles")
plt.ylabel("Recovery rate")
print(
    f"{round((agent.decoding_errors / agent.total_decode) * 100, 2)}% ({agent.decoding_errors}/{agent.total_decode}) decoding errors"
)
