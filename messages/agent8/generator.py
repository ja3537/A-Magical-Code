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
    for n in range(5):
        l = randint(2, 5)
        print(generate(l))
