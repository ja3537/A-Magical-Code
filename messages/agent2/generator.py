from random import *
import string
    
def generate(numMessages, seedNum, w=False):
        if w:
            f = open('airportExMessages.txt', 'w')
        seed(seedNum)
        for m in range(numMessages):
            file = open('airportcodes.txt', 'r')
            content = file.readlines()
            month = randint(1,12)
            day = randint(1,28)
            year = randint(2023,2025)
            airport = randint(0,2018)
            airportCode = content[airport]
            airportCode = airportCode[:-1]
            if month < 10:
                month = '0' + str(month)
            if day < 10:
                day = '0' + str(day)
            N = 4
            res = ''.join(choice(string.ascii_uppercase + string.digits)
                        for i in range(N))
            message = airportCode + ' ' + res + ' ' + str(month) + str(day) + str(year) 
            if w:
                f.write(message+'\n')
        if w:
            f.close()
    
if __name__ == "__main__":
    generate(20, 1, True)
