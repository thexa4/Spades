import random
import torch

from i_player import IPlayer
from torchmax.gamestate import GameState
from torchmax.models.fullplayer import FullPlayer
from torchmax.models.fullplayer2 import FullPlayer2
from torchmax.models.fullplayer3 import FullPlayer3
from card import Card

class TorchPlayer(IPlayer):
	""""Real AI"""

	def __init__(self, path):
		if path.endswith('.pt'):
			self.model = FullPlayer.load(path, 0, device='cpu')[0]
		elif path.endswith('.pt2'):
			self.model = FullPlayer2.load(path, 0, device='cpu')[0]
		elif path.endswith('.pt3'):
			self.model = FullPlayer3.load(path, 0, device='cpu')[0]
			self.model.eval()
		else:
			print("Unknown model type")
			crash()
		self.team0_score = 0
		self.team1_score = 0

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
            cards_played_me=torch.zeros(1, 13, 52),
            cards_played_teammate=torch.zeros(1, 13, 52),
            cards_played_left=torch.zeros(1, 13, 52),
            cards_played_right=torch.zeros(1, 13, 52),
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
		self.gamestate.team_score_mine[0, 0] = self.team0_score // 10
		self.gamestate.team_score_mine[0, 1] = self.team0_score % 10
		self.gamestate.team_score_other[0, 0] = self.team1_score // 10
		self.gamestate.team_score_other[0, 1] = self.team1_score % 10
		

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
		my_bid = result["bids"].item()
		if my_bid > max_bid:
			my_bid = max_bid
		
		#expected_score = result['own_score_prediction'][0, 0].item() * 10 + result['own_score_prediction'][0, 1].item()
		#print(f"I expect to get this score delta: {round(expected_score)}")

		self.gamestate.bid_mine[0] = my_bid
		self.gamestate.team_bid[0] = self.gamestate.team_bid[0] + my_bid

		return my_bid

	def announce_bids(self, bids):
		team0_bid = 0
		team1_bid = 0

		if bids[0] == 'B' or bids[0] == 'N':
			self.gamestate.bid_mine[0] = 0
		else:
			self.gamestate.bid_mine[0] = bids[0]
			team0_bid = team0_bid + bids[0]

		if bids[1] == 'B' or bids[1] == 'N':
			self.gamestate.bid_right[0] = 0
		else:
			self.gamestate.bid_right[0] = bids[1]
			team1_bid = team1_bid + bids[1]
		
		if bids[2] == 'B' or bids[2] == 'N':
			self.gamestate.bid_teammate[0] = 0
		else:
			self.gamestate.bid_teammate[0] = bids[2]
			team0_bid = team0_bid + bids[2]

		if bids[3] == 'B' or bids[3] == 'N':
			self.gamestate.bid_left[0] = 0
		else:
			self.gamestate.bid_left[0] = bids[3]
			team1_bid = team1_bid + bids[3]
		
		self.gamestate.team_bid[0] = team0_bid
		self.gamestate.other_team_bid[0] = team0_bid

		self.round_number = 0
		

	def play_card(self, trick, valid_cards):
		self.gamestate.hand_number[0] = self.round_number
		if trick[3] != None:
			if trick[3].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[3])
			self.gamestate.cards_played_left[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]
		
		if trick[1] != None:
			if trick[1].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[1])
			self.gamestate.cards_played_right[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]

		if trick[2] != None:
			if trick[2].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[2])
			self.gamestate.cards_played_teammate[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]
		
		allowed_cards = torch.tensor([[0 for i in range(52)]])
		for card in valid_cards:
			allowed_cards[0, int(card)] = 1

		torch_result = self.model.play(self.gamestate, allowed_cards)
		result = [x for x in valid_cards if int(x) == torch_result['cards'].item()][0]

		return result
	
	def announce_trick(self, trick):
		if trick[3] != None:
			if trick[3].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[3])
			self.gamestate.cards_played_left[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]
			if trick.get_winner() == 3:
				self.gamestate.hands_won_left[0, self.round_number] = 1
				self.gamestate.trick_wins_left = self.gamestate.trick_wins_left + 1
		
		if trick[0] != None:
			if trick[0].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[0])
			self.gamestate.cards_played_me[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]
			if trick.get_winner() == 0:
				self.gamestate.hands_won_mine[0, self.round_number] = 1
				self.gamestate.trick_wins_me = self.gamestate.trick_wins_me + 1
		
		if trick[1] != None:
			if trick[1].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[1])
			self.gamestate.cards_played_right[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]
			if trick.get_winner() == 1:
				self.gamestate.hands_won_right[0, self.round_number] = 1
				self.gamestate.trick_wins_right = self.gamestate.trick_wins_right + 1
			
		if trick[2] != None:
			if trick[2].suit_id == 0:
				self.gamestate.spades_played[0] = 1

			cardid = int(trick[2])
			self.gamestate.cards_played_teammate[0, self.round_number, cardid] = 1
			self.gamestate.cards_seen[0, cardid]
			if trick.get_winner() == 2:
				self.gamestate.hands_won_teammate[0, self.round_number] = 1
				self.gamestate.trick_wins_teammate = self.gamestate.trick_wins_teammate + 1
		
		self.round_number = self.round_number + 1

	def offer_blind_nill(self, bids):
		return False

	def receive_blind_nill_cards(self, cards):
		self.hand += cards

	def request_blind_nill_cards(self):
		offered_cards = self.hand[-2:]
		self.hand = self.hand[:-2]
		return offered_cards

	def announce_score(self, score):
		self.team0_score = score[0]
		self.team1_score = score[1]