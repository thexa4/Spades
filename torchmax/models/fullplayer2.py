import copy
import torch
import random
from torch import nn
try:
    from models.hand_embedding import HandEmbeddingNet
except:
    from torchmax.models.hand_embedding import HandEmbeddingNet

class HandNet(torch.nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes
        
        layers = []
        current_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_size, layer_size))
            current_size = layer_size
            layers.append(nn.ReLU())

        self.linear_relu_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class BidNet(torch.nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes
        
        layers = []
        current_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_size, layer_size))
            current_size = layer_size
            layers.append(nn.ReLU())
        layers.append(nn.Linear(current_size, 13 * 4))

        self.linear_relu_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class CardNet(torch.nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes

        layers = []
        current_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_size, layer_size))
            current_size = layer_size
            layers.append(nn.ReLU())
        layers.append(nn.Linear(current_size, 52 * 4))
        
        self.linear_relu_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class FullPlayer2:
    BID_INPUT_SIZE=93
    CARD_INPUT_SIZE=2984

    def __init__(self, temperature, hand_layout, bid_layout, card_layout, embedding_path=None, embedding_net=None, device=None):
        self.temperature = min(100, max(temperature, 0.01))
        self.hand_layout = hand_layout
        self.bid_layout = bid_layout
        self.card_layout = card_layout
        self.embedding_path = embedding_path
        self.embedding_net = embedding_net
        if hand_layout == None:
            if embedding_net == None:
                self.handnet = HandEmbeddingNet.load(embedding_path)[0].to(device)
            else:
                self.handnet = embedding_net.to(device)
            handsize = self.handnet.output_size
        else:
            self.handnet = HandNet(52, hand_layout).to(device)
            handsize = hand_layout[-1]
        self.bidnet = BidNet(self.BID_INPUT_SIZE + handsize, bid_layout).to(device)
        self.cardnet = CardNet(self.CARD_INPUT_SIZE + handsize, card_layout).to(device)
        self.device = device
        self.optim_state = None

    def with_temperature(self, temperature):
        result = FullPlayer2(temperature, self.hand_layout, self.bid_layout, self.card_layout, self.embedding_path, self.embedding_net, device=self.device)
        result.handnet = self.handnet
        result.bidnet = self.bidnet
        result.cardnet = self.cardnet
        return result

    def parameters(self):
        return list(self.handnet.parameters()) + list(self.bidnet.parameters()) + list(self.cardnet.parameters())
    
    def bid(self, gamestate, prediction_bid_overrides=None):
        hand_summary = self.handnet(gamestate.my_hand.float())

        stack_items = [
            hand_summary,
            #gamestate.spades_played,
            torch.nn.functional.one_hot(gamestate.team_bid.long(), 14),
            torch.nn.functional.one_hot(gamestate.other_team_bid.long(), 14),
            #torch.nn.functional.one_hot(gamestate.bid_mine.long(), 13),
            torch.nn.functional.one_hot(gamestate.bid_teammate.long(), 14),
            torch.nn.functional.one_hot(gamestate.bid_left.long(), 14),
            torch.nn.functional.one_hot(gamestate.bid_right.long(), 14),
            #gamestate.cards_played_me,
            #gamestate.cards_played_teammate,
            #gamestate.cards_played_left,
            #gamestate.cards_played_right,
            gamestate.team_score_mine[:, 0:1] / 50,
            torch.nn.functional.one_hot(gamestate.team_score_mine[:, 1].long(), 10),
            gamestate.team_score_other[:, 0:1] / 50,
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

        bid_data = self.bidnet(input_data)
        own_scores = torch.reshape(bid_data[:, 0:26], (-1, 13, 2))
        other_scores = torch.reshape(bid_data[:, 26:52], (-1, 13, 2))

        score_delta = torch.tanh(own_scores[:, :, 0]) - torch.tanh(other_scores[:, :, 0])
        score_softmax = torch.nn.functional.softmax(score_delta / self.temperature, 1)
        
        chosen_bids = torch.multinomial(score_softmax, 1).squeeze()
        if prediction_bid_overrides != None:
            chosen_bids = prediction_bid_overrides.to(device=self.device)

        own_raw_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_bids, :]
        own_score_predictions_major = torch.tanh(own_raw_score_predictions[:, 0]) * 30
        own_score_predictions_minor = torch.sigmoid(own_raw_score_predictions[:, 1]) * 12 - 1
        own_score_predictions = torch.stack([own_score_predictions_major, own_score_predictions_minor], 1)

        other_raw_score_predictions = other_scores[torch.arange(other_scores.size(0)), chosen_bids, :]
        other_score_predictions_major = torch.tanh(other_raw_score_predictions[:, 0]) * 30
        other_score_predictions_minor = torch.sigmoid(other_raw_score_predictions[:, 1]) * 12 - 1
        other_score_predictions = torch.stack([other_score_predictions_major, other_score_predictions_minor], 1)

        return {
            "bids": chosen_bids,
            "own_score_prediction": own_score_predictions,
            "other_score_prediction": other_score_predictions,
        }

    def play(self, gamestate, allowed_cards, prediction_indexes_overrides=None):
        hand_summary = self.handnet(gamestate.my_hand.float())

        stack_items = [
            hand_summary,
            gamestate.spades_played.unsqueeze(1),
            torch.nn.functional.one_hot(gamestate.team_bid.long(), 14),
            torch.nn.functional.one_hot(gamestate.other_team_bid.long(), 14),
            torch.nn.functional.one_hot(gamestate.bid_mine.long(), 14),
            torch.nn.functional.one_hot(gamestate.bid_teammate.long(), 14),
            torch.nn.functional.one_hot(gamestate.bid_left.long(), 14),
            torch.nn.functional.one_hot(gamestate.bid_right.long(), 14),
            gamestate.cards_played_me.reshape((-1, 676)),
            gamestate.cards_played_teammate.reshape((-1, 676)),
            gamestate.cards_played_left.reshape((-1, 676)),
            gamestate.cards_played_right.reshape((-1, 676)),
            gamestate.team_score_mine[:, 0:1] / 50,
            torch.nn.functional.one_hot(gamestate.team_score_mine[:, 1].long(), 10),
            gamestate.team_score_other[:, 0:1] / 50,
            torch.nn.functional.one_hot(gamestate.team_score_other[:, 1].long(), 10),
            gamestate.hands_won_mine / 13,
            gamestate.hands_won_teammate / 13,
            gamestate.hands_won_left / 13,
            gamestate.hands_won_right / 13,
            torch.nn.functional.one_hot(gamestate.players_left.long(), 4),
            torch.nn.functional.one_hot(gamestate.hand_number.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_me.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_teammate.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_left.long(), 13),
            torch.nn.functional.one_hot(gamestate.trick_wins_right.long(), 13),
            gamestate.cards_seen
        ]
        input_data = torch.concat([item.float().to(device=self.device) for item in stack_items], 1)

        card_data = self.cardnet(input_data)
        own_scores = torch.reshape(card_data[:, 0:104], (-1, 52, 2))
        other_scores = torch.reshape(card_data[:, 104:208], (-1, 52, 2))

        score_delta = torch.tanh(own_scores[:, :, 0]) - torch.tanh(other_scores[:, :, 0])
        softmax_exponents = torch.exp(score_delta.double() / self.temperature) * allowed_cards.to(device=self.device)
        #softmax_sum = torch.sum(softmax_exponents, 1)
        #score_softmax = softmax_exponents / torch.outer(softmax_sum, torch.ones((52,), device=self.device))
        
        #if torch.any(torch.isnan(score_softmax)):
        #    print(sum(allowed_cards))
        #    print(score_delta)
        #    print(softmax_exponents)
        #    print(softmax_sum)
        #    print(score_softmax)
        chosen_cards = torch.multinomial(softmax_exponents, 1).squeeze()
        if prediction_indexes_overrides != None:
            chosen_cards = prediction_indexes_overrides.to(device=self.device)

        own_raw_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_cards, :]
        own_score_predictions_major = torch.tanh(own_raw_score_predictions[:, 0]) * 30
        own_score_predictions_minor = torch.sigmoid(own_raw_score_predictions[:, 1]) * 12 - 1
        own_score_predictions = torch.stack([own_score_predictions_major, own_score_predictions_minor], 1)

        other_raw_score_predictions = other_scores[torch.arange(other_scores.size(0)), chosen_cards, :]
        other_score_predictions_major = torch.tanh(other_raw_score_predictions[:, 0]) * 30
        other_score_predictions_minor = torch.sigmoid(other_raw_score_predictions[:, 1]) * 12 - 1
        other_score_predictions = torch.stack([other_score_predictions_major, other_score_predictions_minor], 1)

        return {
            "cards": chosen_cards,
            "own_score_prediction": own_score_predictions,
            "other_score_prediction": other_score_predictions,
        }
    
    def with_device(self, device):
        if self.device == device:
            return self

        result = FullPlayer2(self.temperature, self.hand_layout, self.bid_layout, self.card_layout, device=device)
        result.handnet = copy.deepcopy(self.handnet).to(device)
        result.bidnet = copy.deepcopy(self.bidnet).to(device)
        result.cardnet = copy.deepcopy(self.cardnet).to(device)
        return result

    def replicate(self, devices):
        if len(devices) == 1:
            return [self.with_device(devices[0])]

        self.bidnet.share_memory()
        self.cardnet.share_memory()
        self.handnet.share_memory()
        results = [FullPlayer2(self.temperature, self.hand_layout, self.bid_layout, self.card_layout, device=device) for device in devices]
        handnets = torch.nn.parallel.replicate(self.handnet, devices)
        bidnets = torch.nn.parallel.replicate(self.bidnet, devices)
        cardnets = torch.nn.parallel.replicate(self.cardnet, devices)
        for i in range(len(devices)):
            results[i].handnet = handnets[i]
            results[i].bidnet = bidnets[i]
            results[i].cardnet = cardnets[i]
        return results
    
    def mix(self, other, weight):
        result = FullPlayer2(self.temperature, self.hand_layout, self.bid_layout, self.card_layout, self.embedding_path, self.embedding_net, self.device)
        result.handnet.load_state_dict(self.handnet.state_dict())
        result.bidnet.load_state_dict(self.bidnet.state_dict())
        result.cardnet.load_state_dict(self.cardnet.state_dict())

        my_hand_weights = dict(self.handnet.named_parameters())
        other_hand_weights = dict(other.handnet.named_parameters())

        my_bid_weights = dict(self.bidnet.named_parameters())
        other_bid_weights = dict(other.bidnet.named_parameters())

        my_card_weights = dict(self.cardnet.named_parameters())
        other_card_weights = dict(other.cardnet.named_parameters())

        new_hand_weights = dict(my_hand_weights)
        new_bid_weights = dict(my_bid_weights)
        new_card_weights = dict(my_card_weights)

        for name in my_hand_weights.keys():
            new_hand_weights[name].data.copy_((1 - weight) * my_hand_weights[name].data + weight * other_hand_weights[name].data)
        for name in my_bid_weights.keys():
            new_bid_weights[name].data.copy_((1 - weight) * my_bid_weights[name].data + weight * other_bid_weights[name].data)
        for name in my_card_weights.keys():
            new_card_weights[name].data.copy_((1 - weight) * my_card_weights[name].data + weight * other_card_weights[name].data)
        
        result.handnet.load_state_dict(new_hand_weights)
        result.bidnet.load_state_dict(new_bid_weights)
        result.cardnet.load_state_dict(new_card_weights)

        return result
    
    @staticmethod
    def load(path, temperature, device=None):
        state_dict = torch.load(path, weights_only=False, map_location=device)

        hand_layers = state_dict['hand_layout']
        bid_layers = state_dict['bid_layout']
        card_layers = state_dict['card_layout']

        hand_weights = state_dict['hand_weights']
        bid_weights = state_dict['bid_weights']

        if hand_layers != None:
            result = FullPlayer2(temperature, hand_layout=hand_layers, bid_layout=bid_layers, card_layout=card_layers, device=device)
            card_weights = state_dict['card_weights']
            result.cardnet.load_state_dict(card_weights)
        else:
            embedding_net = HandEmbeddingNet.loadState(state_dict['hand_embedding'], device=device)[0]
            result = FullPlayer2(temperature, hand_layout=None, bid_layout=bid_layers, card_layout=card_layers, embedding_net=embedding_net, device=device)

        result.handnet.load_state_dict(hand_weights)
        result.bidnet.load_state_dict(bid_weights)
        result.optim_state = state_dict.get('optimizer', None)
        return (result, state_dict.get('meta', {}))
    
    def save(self, path, meta=None):
        if self.hand_layout != None:
            crash()
            state_dict = {
                'hand_layout': self.handnet.layer_sizes,
                'bid_layout': self.bidnet.layer_sizes,
                'card_layout': self.cardnet.layer_sizes,
                'hand_weights': self.handnet.state_dict(),
                'bid_weights': self.bidnet.state_dict(),
                'card_weights': self.cardnet.state_dict(),
                'meta': meta,
            }
            if self.optim_state != None:
                state_dict['optimizer'] = self.optim_state
            torch.save(state_dict, path)
        else:
            state_dict = {
                'hand_layout': None,
                'bid_layout': self.bidnet.layer_sizes,
                'card_layout': self.cardnet.layer_sizes,
                'hand_weights': self.handnet.state_dict(),
                'bid_weights': self.bidnet.state_dict(),
                'hand_embedding': self.handnet.saveData({}),
                'meta': meta,
            }
            if self.optim_state != None:
                state_dict['optimizer'] = self.optim_state
            torch.save(state_dict, path)
