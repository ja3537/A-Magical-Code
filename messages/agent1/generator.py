import numpy as np

class Generator:
    def __init__(self,domain:str) -> None:
        self.domain = list(domain)
        self.look_up_dict = {}
        index = 0
        self.min_length = 3
        self.max_length = 12
        self.count = 1000
        for i in self.domain:
            self.look_up_dict[index] = i
            index +=1
        self.largest_num = len(self.domain)-1
    
    def set_min(self,min) -> None:
        self.min_length = min
    
    def set_max(self,max) -> None:
        self.max_length = max
    
    def set_count(self,count) -> None:
        self.count = count
    
    def create_word(self,word_length)->str:
        letters_num =np.random.randint(low = 0, high = self.largest_num, size = word_length)
        word = ""
        word = [self.look_up_dict[num] for num in letters_num]
        result = "".join(word)
        return result

    def create_word_list(self) -> list[str]:
        word_length =np.random.randint(low = self.min_length, high = self.max_length, size = self.count)
        word_list = [self.create_word(i) for i in word_length]
        return word_list
    
domain = " 0123456789"
domain+="abcdefghijklmnopqrstuvwxyz"
domain+= "."
file_name = "example.txt" # change the output file name
random_gen = Generator(domain)
random_gen.set_count(1000) # change the number of words generated
random_gen.set_max(12)# change the max lenght of the word
random_gen.set_min(3)# change the min lenght of the word

f = open(file_name, "w") 
word_list= random_gen.create_word_list()
for word in word_list :
    output =word
    output += "\n"
    f.write(output)
f.close()