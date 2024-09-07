import torch
import numpy as np
import math

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
                 mask=None,
                ):
        if my_hand != None:
            self.my_hand = my_hand.to(torch.uint8).detach()
        else:
            self.my_hand = None

        if spades_played != None:
            self.spades_played = spades_played.to(torch.uint8).detach()
        else:
            self.spades_played = None

        if team_bid != None:
            self.team_bid = team_bid.to(torch.uint8).detach()
        else:
            self.team_bid = None

        if other_team_bid != None:
            self.other_team_bid = other_team_bid.to(torch.uint8).detach()
        else:
            self.other_team_bid = None

        if bid_mine != None:
            self.bid_mine = bid_mine.to(torch.uint8).detach()
        else:
            self.bid_mine = None

        if bid_teammate != None:
            self.bid_teammate = bid_teammate.to(torch.uint8).detach()
        else:
            self.bid_teammate = None

        if bid_left != None:
            self.bid_left = bid_left.to(torch.uint8).detach()
        else:
            self.bid_left = None

        if bid_right != None:
            self.bid_right = bid_right.to(torch.uint8).detach()
        else:
            self.bid_right = None

        if cards_played_me != None:
            self.cards_played_me = cards_played_me.to(torch.uint8).detach()
        else:
            self.cards_played_me = None

        if cards_played_teammate != None:
            self.cards_played_teammate = cards_played_teammate.to(torch.uint8).detach()
        else:
            self.cards_played_teammate = None

        if cards_played_left != None:
            self.cards_played_left = cards_played_left.to(torch.uint8).detach()
        else:
            self.cards_played_left = None

        if cards_played_right != None:
            self.cards_played_right = cards_played_right.to(torch.uint8).detach()
        else:
            self.cards_played_right = None

        if team_score_mine != None:
            self.team_score_mine = team_score_mine.to(torch.int16).detach()
        else:
            self.team_score_mine = None

        if team_score_other != None:
            self.team_score_other = team_score_other.to(torch.int16).detach()
        else:
            self.team_score_other = None

        if hands_won_mine != None:
            self.hands_won_mine = hands_won_mine.to(torch.uint8).detach()
        else:
            self.hands_won_mine = None

        if hands_won_teammate != None:
            self.hands_won_teammate = hands_won_teammate.to(torch.uint8).detach()
        else:
            self.hands_won_teammate = None

        if hands_won_left != None:
            self.hands_won_left = hands_won_left.to(torch.uint8).detach()
        else:
            self.hands_won_left = None

        if hands_won_right != None:
            self.hands_won_right = hands_won_right.to(torch.uint8).detach()
        else:
            self.hands_won_right = None

        if players_left != None:
            self.players_left = players_left.to(torch.uint8).detach()
        else:
            self.players_left = None

        if hand_number != None:
            self.hand_number = hand_number.to(torch.uint8).detach()
        else:
            self.hand_number = None

        if trick_wins_me != None:
            self.trick_wins_me = trick_wins_me.to(torch.uint8).detach()
        else:
            self.trick_wins_me = None

        if trick_wins_teammate != None:
            self.trick_wins_teammate = trick_wins_teammate.to(torch.uint8).detach()
        else:
            self.trick_wins_teammate = None

        if trick_wins_left != None:
            self.trick_wins_left = trick_wins_left.to(torch.uint8).detach()
        else:
            self.trick_wins_left = None
        
        if trick_wins_right != None:
            self.trick_wins_right = trick_wins_right.to(torch.uint8).detach()
        else:
            self.trick_wins_right = None
        
        if cards_seen != None:
            self.cards_seen = cards_seen.to(torch.uint8).detach()
        else:
            self.cards_seen = None

        if mask != None:
            self.mask = mask.detach()
        else:
            self.mask = None
    
    def fields(self):
        return {
            "my_hand": self.my_hand,
            "spades_played": self.spades_played,
            "team_bid": self.team_bid,
            "other_team_bid": self.other_team_bid,
            "bid_mine": self.bid_mine,
            "bid_teammate": self.bid_teammate,
            "bid_left": self.bid_left,
            "bid_right": self.bid_right,
            "cards_played_me": self.cards_played_me,
            "cards_played_teammate": self.cards_played_teammate,
            self.cards_played_left,
            self.cards_played_right,
            self.team_score_mine,
            self.team_score_other,
            self.hands_won_mine,
            self.hands_won_teammate,
            self.hands_won_left,
            self.hands_won_right,
            self.players_left,
            self.hand_number,
            self.trick_wins_me,
            self.trick_wins_teammate,
            self.trick_wins_left,
            self.trick_wins_right,
            self.cards_seen,
        }
    
    def combine(self, other_state, mask):
        if self.mask == None:
            raise Exception("Intial mask must be set")
        
        overlap = torch.any(self.mask * mask)
        if overlap.item() == 1:
            raise Exception("Added state overlaps")
            
        combined = self.fields()
        other = other_state.fields()
        for i in range(len(combined)):
            prev = combined[i]
            if prev == None:
                combined[i] = other[i]
            else:
                non_batch_shape = prev.shape[1:]

                if non_batch_shape == ():
                    combined[i] = combined[i] + other[i] * mask
                else:
                    combined[i] = combined[i] + other[i] * torch.reshape(torch.outer(mask, torch.ones(math.prod(non_batch_shape))), (-1,) + non_batch_shape)
            
        return GameState(*combined, mask=torch.maximum(self.mask, mask))

