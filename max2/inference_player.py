from i_player import IPlayer
from max2.gamestate import GameState

class InferencePlayer(IPlayer):
	""""Constitutes the very bare necessity to be called a player"""

	def __init__(self, model):
		self.state = None
		self.round = 0
		self.score = None
		self.model = model
		self.samples = ([[] for i in range(70)], [[] for i in range(2)])

	def give_hand(self, cards):
		self.hand = cards

	def make_bid(self, bids):
		self.state = GameState(self.model)
		self.round = 0
		return self.state.bid(self.hand, bids, self.score)
	def announce_bids(self, bids):
		self.state.store_bids(bids)

	def play_card(self, trick, valid_cards):
		card = self.state.play(self.round, trick, valid_cards)
		self.round = self.round + 1
		return card
	def announce_trick(self, trick):
		self.state.store_trick(trick)

	def announce_score(self, score):
		self.score = score

	def offer_blind_nill(self, bids):
		return False

	def receive_blind_nill_cards(self, cards):
		self.hand += cards

	def request_blind_nill_cards(self):
		offered_cards = self.hand[-2:]
		self.hand = self.hand[:-2]
		return offered_cards