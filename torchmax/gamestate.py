import random
import numpy as np

class GameState:
    def __init__(self,
                 my_hand=None,
                 spades_played=None,
                 team_bid=None,
                 other_team_bid=None,
                 bid_mine=None,
                 bid_teammate=None,
                 bid_left=None,
                 bid_right=None,
                 cards_played_me=None,
                 cards_played_teammate=None,
                 cards_played_left=None,
                 cards_played_right=None,
                 team_score_mine=None,
                 team_score_other=None,
                 hands_won_mine=None,
                 hands_won_teammate=None,
                 hands_won_left=None,
                 hands_won_right=None,
                 players_left=None,
                 hand_number=None,
                 trick_wins_me=None,
                 trick_wins_teammate=None,
                 trick_wins_left=None,
                 trick_wins_right=None,
                 cards_seen=None,
                ):
        self.my_hand = my_hand.detach()
        self.spades_played = spades_played.detach()
        self.team_bid = team_bid.detach()
        self.other_team_bid = other_team_bid.detach()
        self.bid_mine = bid_mine.detach()
        self.bid_teammate = bid_teammate.detach()
        self.bid_left = bid_left.detach()
        self.bid_right = bid_right.detach()
        self.cards_played_me = cards_played_me.detach()
        self.cards_played_teammate = cards_played_teammate.detach()
        self.cards_played_left = cards_played_left.detach()
        self.cards_played_right = cards_played_right.detach()
        self.team_score_mine = team_score_mine.detach()
        self.team_score_other = team_score_other.detach()
        self.hands_won_mine = hands_won_mine.detach()
        self.hands_won_teammate = hands_won_teammate.detach()
        self.hands_won_left = hands_won_left.detach()
        self.hands_won_right = hands_won_right.detach()
        self.players_left = players_left.detach()
        self.hand_number = hand_number.detach()
        self.trick_wins_me = trick_wins_me.detach()
        self.trick_wins_teammate = trick_wins_teammate.detach()
        self.trick_wins_left = trick_wins_left.detach()
        self.trick_wins_right = trick_wins_right.detach()
        self.cards_seen = cards_seen.detach()
