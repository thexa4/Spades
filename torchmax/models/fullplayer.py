import torch
import random
from torch import nn

class BidNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 13 * 4)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class CardNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 2560),
            nn.ReLU(),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 52 * 4)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class FullPlayer:
    BID_INPUT_SIZE=80
    CARD_INPUT_SIZE=2958

    def __init__(self, temperature):
        self.temperature = min(100, max(temperature, 0.01))
        self.bidnet = BidNet(self.BID_INPUT_SIZE)
        self.cardnet = CardNet(self.CARD_INPUT_SIZE)

    def with_temperature(self, temperature):
        result = FullPlayer(temperature)
        result.bidnet = self.bidnet
        result.cardnet = self.cardnet
        return result

    def parameters(self):
        return list(self.bidnet.parameters()) + list(self.cardnet.parameters())
    
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
        input_data = torch.concat([item.float() for item in stack_items], 1)

        bid_data = self.bidnet(input_data)
        own_scores = torch.reshape(bid_data[:, 0:26], (-1, 13, 2))
        other_scores = torch.reshape(bid_data[:, 26:52], (-1, 13, 2))

        score_delta = own_scores[:, :, 0] - other_scores[:, :, 0]
        score_softmax = torch.nn.functional.softmax(score_delta / self.temperature, 1)
        
        chosen_bids = torch.multinomial(score_softmax, 1).squeeze()
        if prediction_bid_overrides != None:
            chosen_bids = prediction_bid_overrides

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
        input_data = torch.concat([item.float() for item in stack_items], 1)
        card_data = self.cardnet(input_data)
        own_scores = torch.reshape(card_data[:, 0:104], (-1, 52, 2))
        other_scores = torch.reshape(card_data[:, 104:208], (-1, 52, 2))

        score_delta = own_scores[:, :, 0] - other_scores[:, :, 0]
        softmax_exponents = torch.exp(score_delta.double() / self.temperature) * allowed_cards
        softmax_sum = torch.sum(softmax_exponents, 1)
        score_softmax = softmax_exponents / torch.outer(softmax_sum, torch.ones((52,)))
        #for i in range(score_delta.size(0)):
        #    if torch.any(torch.isnan(score_softmax[i, :])):
        #        print('NaN')
        #        print(allowed_cards)
        #        print(score_delta[i, :] / self.temperature)
        #        print(softmax_exponents[i, :])
        #        print(softmax_sum[i])
        #        print(score_softmax[i, :])
        #        crash()
        
        chosen_cards = torch.multinomial(score_softmax, 1).squeeze()
        if prediction_indexes_overrides != None:
            chosen_cards = prediction_indexes_overrides

        own_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_cards, :] * 20
        other_score_predictions = other_scores[torch.arange(other_scores.size(0)), chosen_cards, :] * 20

        return {
            "cards": chosen_cards,
            "own_score_prediction": own_score_predictions,
            "other_score_prediction": other_score_predictions,
        }
    
    @staticmethod
    def load(path, temperature, device='cuda'):
        state_dict = torch.load(path, weights_only=False, map_location=device)
        bid_weights = state_dict['bid_weights']
        card_weights = state_dict['card_weights']
        
        result = FullPlayer(temperature)
        result.bidnet.load_state_dict(bid_weights)
        result.cardnet.load_state_dict(card_weights)
        return result
    
    def save(self, path):
        state_dict = {
            'bid_weights': self.bidnet.state_dict(),
            'card_weights': self.cardnet.state_dict()
        }
        torch.save(state_dict, path)