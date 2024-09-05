import torch

class BrainDead:
    def __init__(self):
        pass

    def parameters(self):
        return []
    
    def memory_size(self):
        return 1

    def bid(self, gamestate, prediction_bid_overrides=None):
        batchsize = gamestate.my_hand.shape[0]
        delta = torch.stack([torch.full((batchsize,), 8), torch.full((batchsize,), 0)])

        return {
            "memory": torch.tensor(42),
            "bids": torch.full((batchsize,), 3),
            "own_score_prediction": gamestate.team_score_mine + delta,
            "other_score_prediction": gamestate.team_score_other,
        }

    def play(self, gamestate, memory, allowed_cards, prediction_indexes_overrides=None):
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
            "memory": torch.tensor(42),
            "cards": highest_card_indexes,
            "own_score_prediction": gamestate.team_score_mine + delta,
            "other_score_prediction": gamestate.team_score_other,
        }