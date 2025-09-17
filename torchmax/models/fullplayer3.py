import copy
import torch
import math
from torch import nn
try:
    from models.hand_embedding import HandEmbeddingNet
    from models.played_embedding import PlayedEmbeddingNet
except:
    from torchmax.models.hand_embedding import HandEmbeddingNet
    from torchmax.models.played_embedding import PlayedEmbeddingNet


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
            layers.append(nn.BatchNorm1d(current_size))
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
            layers.append(nn.BatchNorm1d(current_size))
        layers.append(nn.Linear(current_size, 52 * 6))
        
        self.linear_relu_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class FullPlayer3:
    BID_INPUT_SIZE=80
    CARD_INPUT_SIZE=217

    def __init__(self, temperature, caution, bid_layout, card_layout, hand_embedding_path=None, hand_embedding_net=None, played_embedding_path=None, played_embedding_net=None, device=None):
        self.temperature = min(100, max(temperature, 0.01))
        self.bid_layout = bid_layout
        self.card_layout = card_layout
        self.caution = caution

        self.hand_embedding_path = hand_embedding_path
        self.hand_embedding_net = hand_embedding_net
        if hand_embedding_net == None:
            self.handnet = HandEmbeddingNet.load(hand_embedding_path)[0].to(device)
        else:
            self.handnet = hand_embedding_net.to(device)
        
        self.played_embedding_path = played_embedding_path
        self.played_embedding_net = played_embedding_net
        if played_embedding_net == None:
            self.playednet = PlayedEmbeddingNet.load(played_embedding_path)[0].to(device)
        else:
            self.playednet = played_embedding_net.to(device)
        
        self.bidnet = BidNet(self.BID_INPUT_SIZE + self.handnet.output_size, bid_layout).to(device)
        self.cardnet = CardNet(self.CARD_INPUT_SIZE + 2 * self.handnet.output_size + 4 * self.playednet.output_size, card_layout).to(device)
        self.device = device
        self.optim_state = None
        self.debug = False
        self.bid_encoding_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ], device=device)
        self.bid_encoding_matrix.requires_grad_(False)
        self.player_embedding_matrix = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ], device=device)
        self.player_embedding_matrix.requires_grad_(False)

    def with_temperature(self, temperature):
        result = FullPlayer3(temperature, self.caution, self.bid_layout, self.card_layout, self.hand_embedding_path, self.hand_embedding_net, self.played_embedding_path, self.played_embedding_net, device=self.device)
        result.handnet = self.handnet
        result.bidnet = self.bidnet
        result.cardnet = self.cardnet
        result.playednet = self.playednet
        return result

    def parameters(self):
        return list(self.handnet.parameters()) + list(self.bidnet.parameters()) + list(self.cardnet.parameters())
    
    def bid(self, gamestate, prediction_bid_overrides=None):
        hand_summary = self.handnet.forward(gamestate.my_hand.float())

        my_score_rotation_x = torch.sin(gamestate.team_score_mine[:, 1].float() / 10.0 * math.pi).unsqueeze(1)
        my_score_rotation_y = torch.cos(gamestate.team_score_mine[:, 1].float() / 10.0 * math.pi).unsqueeze(1)
        other_score_rotation_x = torch.sin(gamestate.team_score_other[:, 1].float() / 10.0 * math.pi).unsqueeze(1)
        other_score_rotation_y = torch.cos(gamestate.team_score_other[:, 1].float() / 10.0 * math.pi).unsqueeze(1)

        player_embedding = torch.nn.functional.embedding(gamestate.players_left.int(), self.player_embedding_matrix)

        stack_items = [
            hand_summary,
            #gamestate.spades_played,
            torch.nn.functional.embedding(gamestate.team_bid.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.other_team_bid.int(), self.bid_encoding_matrix),
            #torch.nn.functional.one_hot(gamestate.bid_mine.long(), 13),
            torch.nn.functional.embedding(gamestate.bid_teammate.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.bid_left.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.bid_right.int(), self.bid_encoding_matrix),
            #gamestate.cards_played_me,
            #gamestate.cards_played_teammate,
            #gamestate.cards_played_left,
            #gamestate.cards_played_right,
            gamestate.team_score_mine[:, 0:1] / 50,
            my_score_rotation_x,
            my_score_rotation_y,
            gamestate.team_score_other[:, 0:1] / 50,
            other_score_rotation_x,
            other_score_rotation_y,
            #gamestate.hands_won_mine,
            #gamestate.hands_won_teammate,
            #gamestate.hands_won_left,
            #gamestate.hands_won_right,
            player_embedding,
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

        expected_own_score = torch.tanh(own_scores[:, :, 0]) - self.caution * torch.sigmoid(own_scores[:, :, 1])
        expected_other_score = torch.tanh(other_scores[:, :, 0]) + self.caution * torch.sigmoid(other_scores[:, :, 1])

        other_interest = 1.0 / (self.temperature + 1.0) * 0.05
        score_delta = expected_own_score - expected_other_score * other_interest
        score_softmax = torch.nn.functional.softmax(score_delta / self.temperature, 1)
        
        chosen_bids = torch.multinomial(torch.nan_to_num(score_softmax, neginf=0.0), 1).squeeze()
        if prediction_bid_overrides != None:
            chosen_bids = prediction_bid_overrides.to(device=self.device)

        own_raw_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_bids, :]
        other_raw_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_bids, :]
        
        own_score_predictions_mean = torch.tanh(own_raw_score_predictions[:, 0]) * 30
        other_score_predictions_mean = torch.tanh(other_raw_score_predictions[:, 0]) * 30
        
        own_score_predictions_stddev = torch.sigmoid(own_raw_score_predictions[:, 1]) * 30
        other_score_predictions_stddev = torch.sigmoid(other_raw_score_predictions[:, 1]) * 30

        if self.debug:
            import numpy
            print(numpy.rint(chosen_bids.detach().numpy()))
            print(score_softmax.detach().numpy())
            print(numpy.rint((score_delta * 30).detach().numpy()))

        return {
            "bids": chosen_bids,
            "own_score_mean": own_score_predictions_mean,
            "other_score_mean": other_score_predictions_mean,
            "own_score_stddev": own_score_predictions_stddev,
            "other_score_stddev": other_score_predictions_stddev,
        }

    def play(self, gamestate, allowed_cards, prediction_indexes_overrides=None):
        hand_summary = self.handnet(gamestate.my_hand.float())

        played_summary_me = self.playednet.forward(gamestate.cards_played_me.float())
        played_summary_teammate = self.playednet.forward(gamestate.cards_played_teammate.float())
        played_summary_left = self.playednet.forward(gamestate.cards_played_left.float())
        played_summary_right = self.playednet.forward(gamestate.cards_played_right.float())

        cards_missing_summary = self.handnet((1.0 - gamestate.cards_seen.float()))
        
        my_score_rotation_x = torch.sin(gamestate.team_score_mine[:, 1].float() / 10.0 * math.pi).unsqueeze(1)
        my_score_rotation_y = torch.cos(gamestate.team_score_mine[:, 1].float() / 10.0 * math.pi).unsqueeze(1)
        other_score_rotation_x = torch.sin(gamestate.team_score_other[:, 1].float() / 10.0 * math.pi).unsqueeze(1)
        other_score_rotation_y = torch.cos(gamestate.team_score_other[:, 1].float() / 10.0 * math.pi).unsqueeze(1)

        player_embedding = torch.nn.functional.embedding(gamestate.players_left.int(), self.player_embedding_matrix)

        stack_items = [
            hand_summary,
            gamestate.spades_played.unsqueeze(1),
            torch.nn.functional.embedding(gamestate.team_bid.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.other_team_bid.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.bid_mine.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.bid_teammate.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.bid_left.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.bid_right.int(), self.bid_encoding_matrix),
            played_summary_me,
            played_summary_teammate,
            played_summary_left,
            played_summary_right,
            gamestate.team_score_mine[:, 0:1] / 50,
            my_score_rotation_x,
            my_score_rotation_y,
            gamestate.team_score_other[:, 0:1] / 50,
            other_score_rotation_x,
            other_score_rotation_y,
            gamestate.hands_won_mine.float(),
            gamestate.hands_won_teammate.float(),
            gamestate.hands_won_left.float(),
            gamestate.hands_won_right.float(),
            player_embedding,
            torch.nn.functional.embedding(gamestate.hand_number.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.trick_wins_me.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.trick_wins_teammate.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.trick_wins_left.int(), self.bid_encoding_matrix),
            torch.nn.functional.embedding(gamestate.trick_wins_right.int(), self.bid_encoding_matrix),
            cards_missing_summary
        ]
        input_data = torch.concat([item.float().to(device=self.device) for item in stack_items], 1)

        card_data = self.cardnet(input_data).double()
        own_scores = torch.reshape(card_data[:, 0:156], (-1, 52, 3))
        other_scores = torch.reshape(card_data[:, 156:312], (-1, 52, 3))

        expected_own_score = torch.tanh(own_scores[:, :, 0]) - self.caution * torch.sigmoid(own_scores[:, :, 2])
        expected_other_score = torch.tanh(other_scores[:, :, 0]) + self.caution * torch.sigmoid(other_scores[:, :, 2])

        #print([torch.tanh(own_scores[:, :, 0]), torch.sigmoid(own_scores[:, :, 1])])

        score_delta = expected_own_score - expected_other_score * 0.2

        softmax_exponents = torch.exp(score_delta.double() / self.temperature) * allowed_cards.to(device=self.device)
        
        chosen_cards = torch.multinomial(torch.nan_to_num(softmax_exponents, neginf=0.0), 1).squeeze()
        if prediction_indexes_overrides != None:
            chosen_cards = prediction_indexes_overrides.to(device=self.device)

        own_raw_score_predictions = own_scores[torch.arange(own_scores.size(0)), chosen_cards, :]
        own_score_predictions_major = torch.tanh(own_raw_score_predictions[:, 0]) * 30
        #own_score_predictions_minor = torch.sigmoid(own_raw_score_predictions[:, 1]) * 12 - 1

        other_raw_score_predictions = other_scores[torch.arange(other_scores.size(0)), chosen_cards, :]
        other_score_predictions_major = torch.tanh(other_raw_score_predictions[:, 0]) * 30
        #other_score_predictions_minor = torch.sigmoid(other_raw_score_predictions[:, 1]) * 12 - 1
        
        own_score_predictions_stddev = torch.sigmoid(own_raw_score_predictions[:, 2]) * 30
        other_score_predictions_stddev = torch.sigmoid(other_raw_score_predictions[:, 2]) * 30

        return {
            "cards": chosen_cards,
            "own_score_stddev": own_score_predictions_stddev,
            "other_score_stddev": other_score_predictions_stddev,
            #"own_score_bags": own_score_predictions_minor,
            #"other_score_bags": other_score_predictions_minor,
            "own_score_mean": own_score_predictions_major,
            "other_score_mean": other_score_predictions_major,
        }
    
    def with_device(self, device):
        if self.device == device:
            return self

        result = FullPlayer3(self.temperature, self.caution, self.bid_layout, self.card_layout, self.hand_embedding_path, self.hand_embedding_net, self.played_embedding_path, self.played_embedding_net, device=device)
        result.handnet = copy.deepcopy(self.handnet).to(device)
        result.bidnet = copy.deepcopy(self.bidnet).to(device)
        result.cardnet = copy.deepcopy(self.cardnet).to(device)
        result.playednet = copy.deepcopy(self.playednet).to(device)
        result.bid_encoding_matrix = copy.deepcopy(self.bid_encoding_matrix).to(device)
        result.player_embedding_matrix = copy.deepcopy(self.player_embedding_matrix).to(device)
        return result

    def eval(self):
        self.bidnet.eval()
        self.cardnet.eval()
        self.handnet.eval()
        self.playednet.eval()

    def replicate(self, devices):
        if len(devices) == 1:
            return [self.with_device(devices[0])]

        self.bidnet.share_memory()
        self.cardnet.share_memory()
        self.handnet.share_memory()
        self.playednet.share_memory()
        results = [FullPlayer3(self.temperature, self.caution, self.bid_layout, self.card_layout, self.hand_embedding_path, self.hand_embedding_net, self.played_embedding_path, self.played_embedding_net, device=device) for device in devices]
        handnets = torch.nn.parallel.replicate(self.handnet.to(device=devices[0]), devices)
        bidnets = torch.nn.parallel.replicate(self.bidnet.to(device=devices[0]), devices)
        cardnets = torch.nn.parallel.replicate(self.cardnet.to(device=devices[0]), devices)
        playednets = torch.nn.parallel.replicate(self.playednet.to(device=devices[0]), devices)
        for i in range(len(devices)):
            results[i].handnet = handnets[i]
            results[i].bidnet = bidnets[i]
            results[i].cardnet = cardnets[i]
            results[i].playednet = playednets[i]
        return results
    
    def mix(self, other, weight):
        result = FullPlayer3(self.temperature, self.caution, self.bid_layout, self.card_layout, self.hand_embedding_path, self.hand_embedding_net, self.played_embedding_path, self.played_embedding_net, device=self.device)
        result.bidnet.load_state_dict(self.bidnet.state_dict())
        result.cardnet.load_state_dict(self.cardnet.state_dict())
        result.handnet.load_state_dict(self.handnet.state_dict())
        result.playednet.load_state_dict(self.playednet.state_dict())

        my_hand_weights = dict(self.handnet.state_dict())
        other_hand_weights = dict(other.handnet.state_dict())

        my_bid_weights = dict(self.bidnet.state_dict())
        other_bid_weights = dict(other.bidnet.state_dict())

        my_card_weights = dict(self.cardnet.state_dict())
        other_card_weights = dict(other.cardnet.state_dict())
        
        my_played_weights = dict(self.playednet.state_dict())
        other_played_weights = dict(other.playednet.state_dict())

        new_hand_weights = dict(my_hand_weights)
        new_bid_weights = dict(my_bid_weights)
        new_card_weights = dict(my_card_weights)
        new_played_weights = dict(my_played_weights)

        for name in my_bid_weights.keys():
            new_bid_weights[name].data.copy_((1 - weight) * my_bid_weights[name].data + weight * other_bid_weights[name].data)
        for name in my_card_weights.keys():
            new_card_weights[name].data.copy_((1 - weight) * my_card_weights[name].data + weight * other_card_weights[name].data)
        for name in my_hand_weights.keys():
            new_hand_weights[name].data.copy_((1 - weight) * my_hand_weights[name].data + weight * other_hand_weights[name].data)
        for name in my_played_weights.keys():
            new_played_weights[name].data.copy_((1 - weight) * my_played_weights[name].data + weight * other_played_weights[name].data)

        result.handnet.load_state_dict(new_hand_weights)
        result.bidnet.load_state_dict(new_bid_weights)
        result.cardnet.load_state_dict(new_card_weights)
        result.playednet.load_state_dict(new_played_weights)

        return result.with_device(self.device)

    def detach_handnet(self):
        for param in self.handnet.parameters():
            param.requires_grad_(False)

    def detach_playednet(self):
        for param in self.playednet.parameters():
            param.requires_grad_(False)
    
    @staticmethod
    def load(path, temperature, device=None):
        state_dict = torch.load(path, weights_only=False, map_location=device)

        bid_layout = state_dict['bid_layout']
        card_layout = state_dict['card_layout']

        hand_weights = state_dict['hand_weights']
        bid_weights = state_dict['bid_weights']
        caution = state_dict['caution']

        hand_embedding_net = HandEmbeddingNet.loadState(state_dict['hand_embedding'], device=device)[0]
        played_embedding_net = PlayedEmbeddingNet.loadState(state_dict['played_embedding'], device=device)[0]
        result = FullPlayer3(temperature, caution, bid_layout, card_layout, hand_embedding_path=None, hand_embedding_net=hand_embedding_net, played_embedding_path=None, played_embedding_net=played_embedding_net, device=device)

        result.handnet.load_state_dict(hand_weights)
        result.bidnet.load_state_dict(bid_weights)
        result.optim_state = state_dict.get('optimizer', None)
        return (result, state_dict.get('meta', {}))
    
    def save(self, path, meta=None):
        state_dict = {
            'caution': self.caution,
            'bid_layout': self.bidnet.layer_sizes,
            'card_layout': self.cardnet.layer_sizes,
            'hand_weights': self.handnet.state_dict(),
            'bid_weights': self.bidnet.state_dict(),
            'hand_embedding': self.handnet.saveData({}),
            'played_embedding': self.playednet.saveData({}),
            'meta': meta,
        }
        if self.optim_state != None:
            state_dict['optimizer'] = self.optim_state
        torch.save(state_dict, path)
