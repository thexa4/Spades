from i_player import IPlayer


class BraindeadPlayer(IPlayer):
	""""Constitutes the very bare necessity to be called a player"""

	def give_hand(self, cards):
		self.hand = cards

	def make_bid(self, bids):
		max_bid = 13
		if 2 in bids:
			if bids[2] != 'N' and bids[2] != 'B':
				max_bid = 13 - bids[2]

		return min(3, max_bid)

	def play_card(self, trick, valid_cards):
		return valid_cards.pop()

	def offer_blind_nill(self, bids):
		return False

	def receive_blind_nill_cards(self, cards):
		self.hand += cards

	def request_blind_nill_cards(self):
		offered_cards = self.hand[-2:]
		self.hand = self.hand[:-2]
		return offered_cards
