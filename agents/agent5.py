class Agent:
    def str_to_bin(self, msg) -> list[str]:
        """
        str_to_bin transforms string from ascii character to binary
        returns a list of bytes
        """
        encoded_bin = []
        for ch in msg:
            # char to ascii key
            ascii_rep = ord(ch)
            # ascii to binary
            binary_rep = bin(ascii_rep)[2:]
            # padding = ['0'] * (8 - len(binary_rep)) if len(binary_rep) < 8 else []
            encoded_bin.append(binary_rep.zfill(8))
        return encoded_bin

    def bin_to_str(self, bytes) -> str:
        """
        bin_to_str transforms a list of bytes to string
        returns a string
        """
        decoded_str = ''
        for byte in bytes:
            # converts binary to decimal then maps decimal to ascii
            ch = chr(int(byte, 2))
            decoded_str += ch
        return decoded_str
        

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
        
        print('msg passed to encoder bytes: ', self.str_to_bin(message))
        print('msg in str from decoded bytes: ', self.bin_to_str(self.str_to_bin(message)))

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