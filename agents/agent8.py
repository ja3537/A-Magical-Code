import heapq
from math import factorial, log2
from pprint import pprint
from random import Random
from typing import Callable, Optional


# ================
# Message <-> Bits
# ================
class FreqTree:
    char: Optional[str]
    freq: float
    left: Optional["FreqTree"]
    right: Optional["FreqTree"]

    def __init__(self, char: Optional[str], freq: float):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other: "FreqTree"):
        return self.freq < other.freq

    def __eq__(self, other: object):
        if not isinstance(other, FreqTree):
            return NotImplemented
        return self.freq == other.freq


def make_huffman_encoding(frequencies: dict[str, float]) -> dict[str, str]:
    """
    frequencies is a dictionary mapping from a character in the English alphabet to its frequency.
    Returns a dictionary mapping from a character to its Huffman encoding.
    """
    huffman_encoding: dict[str, str] = {}
    heap: list[FreqTree] = []

    for char, freq in frequencies.items():
        heapq.heappush(heap, FreqTree(char, freq))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        node = FreqTree(None, left.freq + right.freq)
        node.left = left
        node.right = right
        heapq.heappush(heap, node)

    root = heapq.heappop(heap)

    def traverse(node, encoding):
        if node.char:
            huffman_encoding[node.char] = encoding
        else:
            traverse(node.left, encoding + "0")
            traverse(node.right, encoding + "1")

    traverse(root, "")

    return huffman_encoding


def huffman_encode_message(message: str, encoding: dict[str, str]) -> str:
    """
    Encodes a message using the given encoding.
    """
    processed_message = message.replace(" ", "")

    encoded_message = []
    for char in processed_message:
        encoded_message.append(encoding[char])
    return "".join(encoded_message)


def huffman_decode_message(encoded_message: str, encoding: dict[str, str]) -> str:
    """
    Decodes a message using the given encoding.
    """
    decoded_message = ""
    while encoded_message:
        dead = True
        for char, code in encoding.items():
            if encoded_message.startswith(code):
                dead = False
                decoded_message += char
                encoded_message = encoded_message[len(code) :]
        if dead:
            break
    return decoded_message


# ==============
# Bits <-> Cards
# ==============
def bottom_cards_encode(value: int, n: int) -> list[int]:
    if value >= factorial(n):
        raise ValueError(f"{value} is too large to encode in {n} cards!")

    cards = []
    current_value = 0
    digits = list(range(n))
    for i in range(n):
        # First n bins of (n-1)!, then n-1 bins of (n-2)!, and so on bin_width = factorial(n - i) // (n - i)
        bin_width = factorial(n - i) // (n - i)
        # Target value falls into one of these bins
        bin_no = (value - current_value) // bin_width
        # Of the cards still available, ordered by value, select the bin_no'th largest
        card_value = digits[bin_no]
        # Once a card has been used, it can't be used again
        digits = [digit for digit in digits if digit != card_value]
        cards.append(card_value)
        # Move to the bottom of the current bin
        current_value += bin_no * bin_width

    return cards


def bottom_cards_decode(cards: list[int], n: int) -> int:
    cards = [card for card in cards if card < n]
    # Assuming the top card is the first card in the list
    lo = 0
    hi = factorial(n)
    digits = list(range(n))
    for i in range(n):
        card_value = cards[i]
        # Range will always be divisible by the place value
        bin_width = (hi - lo) // (n - i)
        bin_no = digits.index(card_value)
        lo = lo + bin_no * bin_width
        hi = lo + bin_width
        digits = [digit for digit in digits if digit != card_value]

    return lo


def find_c_for_message(bits: str) -> int:
    # In practice, with an 8 bits checksum, this value is always > 5,
    for c in range(1, 52):
        if int(log2(factorial(c))) >= len(bits):
            return c
    return 0


def to_bit_string(value: int) -> str:
    """255 -> 11111111"""
    return bin(value)[2:]


def from_bit_string(bits: str) -> int:
    """11111111 -> 255"""
    return int("0b" + bits, 2)


# ===================
# Checksum Algorithms
# ===================
def quarter_sum_checksum(sent_message: str) -> str:
    k = int(((len(sent_message) / 4) + 1) // 1)

    # Dividing sent message in packets of k bits.
    c1 = sent_message[0:k]
    c2 = sent_message[k : 2 * k]
    c3 = sent_message[2 * k : 3 * k]
    c4 = sent_message[3 * k : 4 * k]

    # Calculating the binary sum of packets
    sum = bin(int(c1, 2) + int(c2, 2) + int(c3, 2) + int(c4, 2))[2:]

    # Adding the overflow bits
    if len(sum) > k:
        x = len(sum) - k
        sum = bin(int(sum[0:x], 2) + int(sum[x:], 2))[2:]
    if len(sum) < k:
        sum = "0" * (k - len(sum)) + sum

    # Calculating the complement of sum
    checksum = ""
    for i in sum:
        if i == "1":
            checksum += "0"
        else:
            checksum += "1"
    return checksum


# Create separate random instance with constant seed
# so that the pearson_table is always the same
random = Random(0)
pearson_table = list(range(256))
random.shuffle(pearson_table)


def pearson_checksum(bits: str) -> str:
    padding = 8 - (len(bits) % 8)
    padded_bits = "0" * padding + bits
    checksum = 0
    for off in range(0, len(padded_bits), 8):
        byte = padded_bits[off : off + 8]
        byte_val = from_bit_string(byte)
        checksum = pearson_table[checksum ^ byte_val]
    checksum_bits = to_bit_string(checksum)
    padded_checksum = pad(checksum_bits, 8)
    return padded_checksum


def check_and_remove(bits: str) -> tuple[bool, str]:
    # Pad in case of very low numbers (leading 0's are trimmed by card encoding)
    message_checksum = pad(bits[-16:-8], 8)
    message_length = from_bit_string(pad(bits[-8:], 8))
    # message_checksum = pad(bits[-8:], 8)
    # message_length = from_bit_string(pad(bits[-16:-8], 8))
    message = bits[:-16]

    # if len(message) > message_length:
    #     return False, ""

    # Pad message to target length with leading 0's
    message = pad(message, message_length, allow_over=True)
    checked_checksum = pearson_checksum(message)
    return checked_checksum == message_checksum, message


def length_byte(bits: str) -> str:
    length_bits = to_bit_string(len(bits))
    return pad(length_bits, 8)


def pad(message: str, length: int, allow_over=False):
    if len(message) > length and not allow_over:
        raise ValueError(
            f"Message to pad is length {len(message)}, longer than target length {length}: {message}"
        )
    needed_padding = length - len(message)
    return "0" * needed_padding + message


# ============
# Multiplexing
# ============
LOWERCASE_HUFFMAN = make_huffman_encoding(
    {
        " ": 1 / (4.7 + 1),
        "a": 0.08167 * (4.7 / 5.7),
        "b": 0.01492 * (4.7 / 5.7),
        "c": 0.02782 * (4.7 / 5.7),
        "d": 0.04253 * (4.7 / 5.7),
        "e": 0.12702 * (4.7 / 5.7),
        "f": 0.02228 * (4.7 / 5.7),
        "g": 0.02015 * (4.7 / 5.7),
        "h": 0.06094 * (4.7 / 5.7),
        "i": 0.06966 * (4.7 / 5.7),
        "j": 0.00153 * (4.7 / 5.7),
        "k": 0.00772 * (4.7 / 5.7),
        "l": 0.04025 * (4.7 / 5.7),
        "m": 0.02406 * (4.7 / 5.7),
        "n": 0.06749 * (4.7 / 5.7),
        "o": 0.07507 * (4.7 / 5.7),
        "p": 0.01929 * (4.7 / 5.7),
        "q": 0.00095 * (4.7 / 5.7),
        "r": 0.05987 * (4.7 / 5.7),
        "s": 0.06327 * (4.7 / 5.7),
        "t": 0.09056 * (4.7 / 5.7),
        "u": 0.02758 * (4.7 / 5.7),
        "v": 0.00978 * (4.7 / 5.7),
        "w": 0.02360 * (4.7 / 5.7),
        "x": 0.00150 * (4.7 / 5.7),
        "y": 0.01974 * (4.7 / 5.7),
        "z": 0.00074 * (4.7 / 5.7),
    }
)
# [(encode, decode)]
# Encoding identifier denotes index in this list
CHARACTER_ENCODINGS: list[tuple[Callable[[str], str], Callable[[str], str]]] = [
    (
        lambda m: huffman_encode_message(m, LOWERCASE_HUFFMAN),
        lambda m: huffman_decode_message(m, LOWERCASE_HUFFMAN),
    )
]


def select_character_encoding(message: str) -> tuple[str, int]:
    """Select the shortest encoding for this message."""
    shortest_encoding = -1
    shortest_encoded: Optional[str] = None
    for encoding_index, (encode, _) in enumerate(CHARACTER_ENCODINGS):
        try:
            candidate_encoding = encode(message)
            if shortest_encoded is None or len(candidate_encoding) < len(
                shortest_encoded
            ):
                shortest_encoded = candidate_encoding
                shortest_encoding = encoding_index
        except ValueError:
            pass

    if shortest_encoded is None:
        raise ValueError(
            f"Could not encode message with any available encodings: {message}"
        )

    return shortest_encoded, shortest_encoding


class Agent:
    def __init__(self):
        # Source for frequencies: http://mathcenter.oxford.emory.edu/site/math125/englishLetterFreqs/
        # the average english word is 4.7 characters long, so we take the probability of a space to be 1/(4.7+1)
        self.frequencies = {
            " ": 1 / (4.7 + 1),
            "a": 0.08167 * (4.7 / 5.7),
            "b": 0.01492 * (4.7 / 5.7),
            "c": 0.02782 * (4.7 / 5.7),
            "d": 0.04253 * (4.7 / 5.7),
            "e": 0.12702 * (4.7 / 5.7),
            "f": 0.02228 * (4.7 / 5.7),
            "g": 0.02015 * (4.7 / 5.7),
            "h": 0.06094 * (4.7 / 5.7),
            "i": 0.06966 * (4.7 / 5.7),
            "j": 0.00153 * (4.7 / 5.7),
            "k": 0.00772 * (4.7 / 5.7),
            "l": 0.04025 * (4.7 / 5.7),
            "m": 0.02406 * (4.7 / 5.7),
            "n": 0.06749 * (4.7 / 5.7),
            "o": 0.07507 * (4.7 / 5.7),
            "p": 0.01929 * (4.7 / 5.7),
            "q": 0.00095 * (4.7 / 5.7),
            "r": 0.05987 * (4.7 / 5.7),
            "s": 0.06327 * (4.7 / 5.7),
            "t": 0.09056 * (4.7 / 5.7),
            "u": 0.02758 * (4.7 / 5.7),
            "v": 0.00978 * (4.7 / 5.7),
            "w": 0.02360 * (4.7 / 5.7),
            "x": 0.00150 * (4.7 / 5.7),
            "y": 0.01974 * (4.7 / 5.7),
            "z": 0.00074 * (4.7 / 5.7),
        }
        self.encoding = make_huffman_encoding(self.frequencies)

    def encode(self, message: str):
        print("Message:", message)
        try:
            encoded, encoding_id = select_character_encoding(message)
        except ValueError as e:
            print(e)
            return list(range(52))

        # message_length = length_byte(encoded)
        # with_length = encoded + message_length
        # checksum = pearson_checksum(with_length)
        # with_checksum = with_length + checksum

        message_length = length_byte(encoded)
        checksum = pearson_checksum(encoded)
        with_checksum = encoded + checksum + message_length
        print("Encoded:", with_checksum)

        c = find_c_for_message(with_checksum)
        encoded = bottom_cards_encode(from_bit_string(with_checksum), c)
        return list(range(c, 52)) + encoded

    def decode(self, deck: list[int]):
        # Minimum 16 bit suffix for checksum + padding
        # log2(9!) > 2 ^ 16
        for c in range(9, 52):
            encoded = [card for card in deck if card < c]
            decoded = to_bit_string(bottom_cards_decode(encoded, c))
            passes_checksum, message = check_and_remove(decoded)
            if passes_checksum:
                out_message = huffman_decode_message(
                    "".join(map(str, message)), self.encoding
                )
                return out_message
        return "NULL"


if __name__ == "__main__":
    agent = Agent()
    pprint(agent.encoding)

    for n in range(1, 52):
        assert (
            bottom_cards_decode(bottom_cards_encode(factorial(n) - 1, n), n)
            == factorial(n) - 1
        )
        assert (
            bottom_cards_decode(bottom_cards_encode(factorial(n) // 2, n), n)
            == factorial(n) // 2
        )
        assert bottom_cards_decode(bottom_cards_encode(0, n), n) == 0
