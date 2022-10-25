class Agent:
    def __init__(self):
        self.stop_card = 51
        self.trash_cards = list(range(32, 51))
        

    def encode(self, message):
        encoded_message = []

        useless_cards = [card for card in range(0, 32) if card not in encoded_message]
        deck = self.trash_cards + useless_cards + [self.stop_card] + encoded_message
        return deck

    def decode(self, deck):
        deck = self.remove_trash_cards(deck)
        deck = self.get_encoded_message(deck)
        return "NULL"


    def remove_trash_cards(self, deck):
        for i in self.trash_cards:
            deck.remove(i)
        return deck

    def get_encoded_message(self, deck):
        deck.index(self.stop_card)
        return deck[deck.index(self.stop_card)+1:]
