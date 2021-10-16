from i_player import IPlayer
from max2.gamestate import GameState

class TrainingPlayer(IPlayer):
	""""Constitutes the very bare necessity to be called a player"""

	def __init__(self, model):
		self.state = None
		self.round = 0
		self.score = None
		self.model = model
		self.samples = []
		self.scored = False

	def give_hand(self, cards):
		self.hand = cards

	def make_bid(self, bids):
		self.scored = False
		self.state = GameState(self.model, training=True)
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

	def offer_blind_nill(self, bids):
		return False

	def receive_blind_nill_cards(self, cards):
		self.hand += cards

	def request_blind_nill_cards(self):
		offered_cards = self.hand[-2:]
		self.hand = self.hand[:-2]
		return offered_cards

	def announce_score(self, score):
		if self.score == None or self.scored:
			self.score = score
			return
		self.scored = True

		d = self.state.store_game(score[0] - self.score[0])
		self.samples.append(d)

		self.score = score