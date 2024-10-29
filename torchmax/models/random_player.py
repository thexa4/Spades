import copy
import torch
import random
from torch import nn

class RandomPlayer:
    BID_INPUT_SIZE=80
    CARD_INPUT_SIZE=2958

    def __init__(self, device=None):
        self.temperature = 1
        self.device = device

    def with_temperature(self, _temperature):
        return self

    def parameters(self):
        return list()
    
    def bid(self, gamestate, prediction_bid_overrides=None):
        stack_items = [
            gamestate.my_hand,
            #gamestate.spades_played,
            gamestate.team_bid.unsqueeze(1),
            gamestate.other_team_bid.unsqueeze(1),
            #gamestate.bid_mine,
            gamestate.bid_teammate.unsqueeze(1),
            gamestate.bid_left.unsqueeze(1),
            gamestate.bid_right.unsqueeze(1),
            #gamestate.cards_played_me,
            #gamestate.cards_played_teammate,
            #gamestate.cards_played_left,
            #gamestate.cards_played_right,
            gamestate.team_score_mine[:, 0:1],
            torch.nn.functional.one_hot(gamestate.team_score_mine[:, 1].long(), 10),
            gamestate.team_score_other[:, 0:1],
            torch.nn.functional.one_hot(gamestate.team_score_other[:, 1].long(), 10),
            #gamestate.hands_won_mine,
            #gamestate.hands_won_teammate,
            #gamestate.hands_won_left,
            #gamestate.hands_won_right,
            gamestate.players_left.unsqueeze(1),
            #gamestate.hand_number,
            #gamestate.trick_wins_me,
            #gamestate.trick_wins_teammate,
            #gamestate.trick_wins_left,
            #gamestate.trick_wins_right,
            #gamestate.cards_seen,
        ]
        input_data = torch.concat([item.float().to(device=self.device) for item in stack_items], 1)

        bid_data = torch.zeros((input_data.size(0), 52,), device=self.device)
        own_scores = torch.reshape(bid_data[:, 0:26], (-1, 13, 2))
        other_scores = torch.reshape(bid_data[:, 26:52], (-1, 13, 2))

        score_delta = own_scores[:, :, 0] - other_scores[:, :, 0]
        score_softmax = torch.nn.functional.softmax(score_delta / self.temperature, 1)
        
        chosen_bids = torch.multinomial(score_softmax, 1).squeeze()
        if prediction_bid_overrides != None:
            chosen_bids = prediction_bid_overrides.to(device=self.device)

        own_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_bids, :] * 20
        other_score_predictions = other_scores[torch.arange(other_scores.size(0)), chosen_bids, :] * 20

        return {
            "bids": chosen_bids,
            "own_score_prediction": own_score_predictions,
            "other_score_prediction": other_score_predictions,
        }

    def play(self, gamestate, allowed_cards, prediction_indexes_overrides=None):
        stack_items = [
            gamestate.my_hand,
            gamestate.spades_played.unsqueeze(1),
            gamestate.team_bid.unsqueeze(1),
            gamestate.other_team_bid.unsqueeze(1),
            gamestate.bid_mine.unsqueeze(1),
            gamestate.bid_teammate.unsqueeze(1),
            gamestate.bid_left.unsqueeze(1),
            gamestate.bid_right.unsqueeze(1),
            gamestate.cards_played_me.reshape((-1, 676)),
            gamestate.cards_played_teammate.reshape((-1, 676)),
            gamestate.cards_played_left.reshape((-1, 676)),
            gamestate.cards_played_right.reshape((-1, 676)),
            gamestate.team_score_mine[:, 0:1],
            torch.nn.functional.one_hot(gamestate.team_score_mine[:, 1].long(), 10),
            gamestate.team_score_other[:, 0:1],
            torch.nn.functional.one_hot(gamestate.team_score_other[:, 1].long(), 10),
            gamestate.hands_won_mine,
            gamestate.hands_won_teammate,
            gamestate.hands_won_left,
            gamestate.hands_won_right,
            torch.nn.functional.one_hot(gamestate.players_left.long(), 4),
            torch.nn.functional.one_hot(gamestate.hand_number.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_me.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_teammate.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_left.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_right.long(), 13),
            gamestate.cards_seen
        ]
        input_data = torch.concat([item.float().to(device=self.device) for item in stack_items], 1)
        card_data = torch.zeros((input_data.size(0), 52 * 4,), device=self.device)
        own_scores = torch.reshape(card_data[:, 0:104], (-1, 52, 2))
        other_scores = torch.reshape(card_data[:, 104:208], (-1, 52, 2))

        score_delta = own_scores[:, :, 0] - other_scores[:, :, 0]
        softmax_exponents = torch.exp(score_delta.double() / self.temperature) * allowed_cards.to(device=self.device)
        softmax_sum = torch.sum(softmax_exponents, 1)
        score_softmax = softmax_exponents / torch.outer(softmax_sum, torch.ones((52,), device=self.device))
        
        if torch.any(torch.isnan(score_softmax)):
            print(sum(allowed_cards))
            print(score_delta)
            print(softmax_exponents)
            print(softmax_sum)
            print(score_softmax)
        chosen_cards = torch.multinomial(score_softmax, 1).squeeze()
        if prediction_indexes_overrides != None:
            chosen_cards = prediction_indexes_overrides.to(device=self.device)

        own_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_cards, :] * 20
        other_score_predictions = other_scores[torch.arange(other_scores.size(0)), chosen_cards, :] * 20

        return {
            "cards": chosen_cards,
            "own_score_prediction": own_score_predictions,
            "other_score_prediction": other_score_predictions,
        }
    
    def with_device(self, device):
        result = RandomPlayer(device=device)
        return result

    def replicate(self, _devices):
        return self
    