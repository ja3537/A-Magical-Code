import os.path as path
from random import choice, randint

BASE_DIR = path.dirname(__file__)
NAMES = [line.strip() for line in open(path.join(BASE_DIR, "names.txt"))]
PLACES = [line.strip() for line in open(path.join(BASE_DIR, "places.txt"))]


def generate(length: int):
    tokens = []
    for _ in range(length):
        domain = choice([NAMES, PLACES])
        token = choice(domain)
        tokens.append(token)
    return " ".join(tokens)


if __name__ == "__main__":
    with open(path.join(path.dirname(__file__), "test.txt"), "w+") as f:
        for n in range(1000):
            l = randint(2, 5)
            f.write(generate(l))
            f.write("\n")
