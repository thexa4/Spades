from i_player import IPlayer
import random

class RandomPlayer(IPlayer):
	""""Constitutes the very bare necessity to be called a player"""

	def give_hand(self, cards):
		self.hand = cards

	def make_bid(self, bids):
		return random.randrange(5)

	def play_card(self, trick, valid_cards):
		return random.choice(valid_cards)

	def offer_blind_nill(self, bids):
		return False

	def receive_blind_nill_cards(self, cards):
		self.hand += cards

	def request_blind_nill_cards(self):
		offered_cards = self.hand[-2:]
		self.hand = self.hand[:-2]
		return offered_cards
