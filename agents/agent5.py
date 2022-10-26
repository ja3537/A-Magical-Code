class Agent:
    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        # 1 start of msg, 26 letters of the alphabet, 10 numbers, and 1 end msg signal

        for i in range(26):
            self.encoder[chr(i + ord('a'))] = i
            self.decoder[i] = chr(i + ord('a'))

        for i in range(10):
            self.encoder[str(i)] = i + 26
            self.decoder[i + 26] = str(i)

        self.encoder['#'] = 36
        self.decoder[36] = '#'
        self.encoder['/'] = 37
        self.decoder[37] = '/'
        

    def encode(self, message):
       # terminal
        msgList = []

        # start '#', end '/'
        message = '#' + message + '/'
        
        encoded = list(range(38, 52)) # red section
        availL = set(range(38))
        
        for x in message:
            msgList.append(self.encoder[x])
            availL.remove(self.encoder[x])

        encoded += list(availL)
        encoded += msgList

        return encoded

    def decode(self, deck):
        msg = ''
        

        in_msg = False
        for x in deck:
            if x == self.encoder['#']:
                in_msg = True
            elif x == self.encoder['/']:
                in_msg = False
            elif in_msg and x in self.decoder:
                msg += self.decoder[x]

        return msg