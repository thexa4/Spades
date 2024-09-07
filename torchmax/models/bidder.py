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

class Bidder:
    BID_INPUT_SIZE=62

    def __init__(self, temperature):
        self.temperature = max(temperature, 0.001)
        self.bidnet = BidNet(self.BID_INPUT_SIZE)

    def with_temperature(self, temperature):
        result = Bidder(temperature)
        result.bidnet = self.bidnet
        return result

    def parameters(self):
        return list(self.bidnet.parameters())
    
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
            gamestate.team_score_mine,
            gamestate.team_score_other,
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
        batchsize = gamestate.my_hand.shape[0]

        values = torch.tensor([[
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, # spades 2-A
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # hearts 2-A
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # clubs: 2-A
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,          # diamonds: 2-A
        ]]).repeat(batchsize, 1)


        highest_card_indexes = torch.argmax(allowed_cards * values, dim=1)
        delta = torch.stack([torch.full((batchsize,), 8), torch.full((batchsize,), 0)])
        
        return {
            "cards": highest_card_indexes,
            "own_score_prediction": delta,
            "other_score_prediction": torch.zeros((batchsize, 2)),
        }
    
    @staticmethod
    def load(path, temperature, device='cuda'):
        state_dict = torch.load(path, weights_only=False, map_location=device)
        weights = state_dict['weights']
        
        result = Bidder(temperature)
        result.bidnet.load_state_dict(weights)
        return result
    
    def save(self, path):
        state_dict = {
            'weights': self.bidnet.state_dict()
        }
        torch.save(state_dict, path)