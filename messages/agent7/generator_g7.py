from pathlib import Path
import random
import enchant

ENGLISH_DICTIONARY = enchant.Dict("en_US")

class Generator_G7():
    def __init__(self, dictionary_path):
        self.dictionary = self.preprocess(dictionary_path)
        
    def preprocess(self, dictionary_path):
        dictionary = set()
        with open(dictionary_path, "r") as f:
            line = f.readline()
            while line:
                line = line.strip()
                if ENGLISH_DICTIONARY.check(line):
                    dictionary.add(line)
                line = f.readline()
        return list(dictionary)

    def generate_msg(self, num_msg, output_path):
        with open(output_path, "w") as f:
            for _ in range(num_msg):
                k = random.randint(1, 6)
                msg = " ".join(random.choices(self.dictionary, k=k))
                f.write(msg + "\n")
        

def main():
    generator = Generator_G7(Path("./messages/agent7/30k.txt"))
    generator.generate_msg(100, Path("./messages/agent7/30k_test.txt"))
    
if __name__ == '__main__':
    main()