import random
import torch

from i_player import IPlayer
from torchmax.gamestate import GameState
from torchmax.models.bidder import Bidder

class TorchPlayer(IPlayer):
	""""Real AI"""

	def __init__(self, path):
		self.model = Bidder.load(path, 0, device='cpu')

	def give_hand(self, cards):
		hand_tensor = torch.zeros((1, 52))
		for card in cards:
			hand_tensor[0, int(card)] = 1

		self.gamestate = GameState(
            my_hand=hand_tensor,
            spades_played=torch.tensor([0]),
            team_bid=torch.tensor([0]),
            other_team_bid=torch.tensor([0]),
            bid_mine=torch.tensor([0]),
            bid_teammate=torch.tensor([0]),
            bid_left=torch.tensor([0]),
            bid_right=torch.tensor([0]),
            cards_played_me=torch.zeros(1, 52),
            cards_played_teammate=torch.zeros(1, 52),
            cards_played_left=torch.zeros(1, 52),
            cards_played_right=torch.zeros(1, 52),
            team_score_mine=torch.zeros(1, 2),
            team_score_other=torch.zeros(1, 2),
            hands_won_mine=torch.zeros(1, 13, dtype=torch.uint8),
            hands_won_teammate=torch.zeros(1, 13, dtype=torch.uint8),
            hands_won_left=torch.zeros(1, 13, dtype=torch.uint8),
            hands_won_right=torch.zeros(1, 13, dtype=torch.uint8),
            players_left=torch.tensor([0]),
            hand_number=torch.tensor([0]),
            trick_wins_me=torch.tensor([0]),
            trick_wins_teammate=torch.tensor([0]),
            trick_wins_left=torch.tensor([0]),
            trick_wins_right=torch.tensor([0]),
            cards_seen=torch.zeros(1, 52),
		)

	def make_bid(self, bids):
		max_bid = 13
		if 2 in bids:
			if bids[2] != 'N' and bids[2] != 'B':
				max_bid = 13 - bids[2]

		players_left = 4
		if bids[1] != None:
			players_left = players_left - 1
			if bids[1] == 'B' or bids[1] == 'N':
				self.gamestate.bid_right[0] = 0
			else:
				self.gamestate.bid_right[0] = bids[1]
				self.gamestate.other_team_bid[0] = self.gamestate.other_team_bid[0] + bids[1]

		if bids[2] != None:
			players_left = players_left - 1
			if bids[2] == 'B' or bids[2] == 'N':
				self.gamestate.bid_teammate[0] = 0
			else:
				self.gamestate.bid_teammate[0] = bids[2]
				self.gamestate.team_bid[0] = self.gamestate.team_bid[0] + bids[2]

		if bids[3] != None:
			players_left = players_left - 1
			if bids[3] == 'B' or bids[3] == 'N':
				self.gamestate.bid_left[0] = 0
			else:
				self.gamestate.bid_left[0] = bids[3]
				self.gamestate.other_team_bid[0] = self.gamestate.other_team_bid[0] + bids[3]

		result = self.model.bid(self.gamestate)
		self.memory = result["memory"]
		my_bid = result["bids"].item()
		if my_bid > max_bid:
			my_bid = max_bid

		self.gamestate.bid_mine[0] = my_bid
		self.gamestate.team_bid[0] = self.gamestate.team_bid[0] + my_bid

		return my_bid

	def announce_bids(self, bids):
		pass

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
