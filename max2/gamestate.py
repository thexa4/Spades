import random
import numpy as np
import tensorflow as tf

class GameState:
    def __init__(self, model, training=False, temperature=1):
        self.bid_state = {'bids': {}, 'hand': [], 'bags': []}
        self.chosen_bid = None
        self.seen = [0] * 52
        self.rounds = [None] * 13
        self.chosen_cards = [None] * 13
        self.tricks = [0] * 4
        self.hand = [0] * 52
        self.orig_bids = None
        self.samples = []
        self.model = model
        self.training = training
        self.temperature = temperature

    def thermometer(self, size, n):
        result = [0] * size
        for i in range(n):
            result[i] = 1
        return result

    def sample(self, options):
        if not self.training:
            return np.argmax(options)

        positive_options = (np.array(options) + 200.0) / 400.0
        exponentized = np.exp(positive_options / self.temperature)
        exp_sum = np.sum(exponentized)

        probabilities = exponentized / exp_sum
        return np.random.choice(len(probabilities), p=probabilities)

    def bid(self, hand, bids, score):
        bidcount = [15,15,15,15]
        for i in range(4):
            if i in bids:
                b = bids[i]
                if b == 'N':
                    bidcount[i] = 0
                elif b == 'B':
                    bidcount[i] = 14
                else:
                    bidcount[i] = b
        self.bid_state['bids'] = [self.thermometer(15, bidcount[i]) for i in range(4)]
        self.bid_state['bags'] = [[0] * 10 for i in range(2)]
        self.bid_state['bags'][0][score[0] % 10] = 1
        self.bid_state['bags'][1][score[1] % 10] = 1

        self.bid_state['hand'] = [0] * 52
        for card in hand:
            self.bid_state['hand'][int(card)] = 1
            self.seen[int(card)] = 1
            self.hand[int(card)] = 1
        
        prediction = self.compute()
        
        max_bid = 13
        if 2 in bids:
            if bids[2] != 'N' and bids[2] != 'B':
                max_bid = 13 - bids[2]

        allowed_bids = prediction['bids'][:(max_bid + 1)]
        self.chosen_bid = self.sample(allowed_bids)
        
        return self.chosen_bid
    
    def store_bids(self, bids):
        self.orig_bids = [0,0,0,0]
        bidcount = [15,15,15,15]
        for i in range(4):
            if i in bids:
                b = bids[i]
                if b == 'N':
                    bidcount[i] = 0
                elif b == 'B':
                    bidcount[i] = 14
                else:
                    bidcount[i] = b
                    self.orig_bids[i] = b
        self.bid_state['bids'] = [self.thermometer(15, bidcount[i]) for i in range(4)]

    def play(self, round, trick, valid_cards):
        played_cards = [[0] * 52 for i in range(3)]
        for player in trick.cards:
            self.seen[int(trick.cards[player])] = 1
            played_cards[player - 1][int(trick.cards[player])] = 1
        
        todo = [[0] * 26 for i in range(4)]
        for i in range(4):
            delta = self.orig_bids[i] - self.tricks[i]
            if delta < 0:
                for j in range(-delta + 1):
                    todo[i][25 - j] = 1
            elif delta > 0:
                for j in range(delta + 1):
                    todo[i][j] = 1


        r = {
            'seen': [0] * 52,
            'hand': self.hand.copy(),
            'played': played_cards,
            'todo': todo,
        }
        
        self.rounds[round] = r
        if len(valid_cards) == 1:
            self.chosen_cards[round] = int(valid_cards[0])
            return valid_cards[0]

        prediction = self.compute()
        allowed_preds = [prediction['round' + str(round)][int(card)] for card in valid_cards]
        chosen = valid_cards[self.sample(allowed_preds)]
        self.chosen_cards[round] = int(chosen)

        return chosen

    def store_trick(self, trick):
        for player in trick.cards:
            self.seen[int(trick.cards[player])] = 1
        self.tricks[trick.get_winner()] += 1

    def empty_round(self):
        return {
            'seen': [0] * 52,
            'hand': [0] * 52,
            'played': [[0] * 52 for i in range(3)],
            'todo': [[0] * 26 for i in range(4)],
        }
    def input(self):
        rounds = [None] * 13
        for i in range(13):
            if self.rounds[i] == None:
                rounds[i] = self.empty_round()
            else:
                rounds[i] = self.rounds[i]
            
        result = {
            'bid_state_bids': np.array(self.bid_state['bids']).flatten(),
            'bid_state_hand': self.bid_state['hand'],
            'bid_state_bags': np.array(self.bid_state['bags']).flatten(),
        }
        for i in range(13):
            roundname = 'round' + str(i) + '_'

            result[roundname + 'seen'] = rounds[i]['seen']
            result[roundname + 'hand'] = rounds[i]['hand']
            result[roundname + 'played'] = np.array(rounds[i]['played']).flatten()
            result[roundname + 'todo'] = np.array(rounds[i]['todo']).flatten()
        
        return result

    def compute(self):
        if self.model == None:
            return {
                'bids': [random.uniform(0, 100) for i in range(14)],
                'round0': [random.uniform(0, 100) for i in range(52)],
                'round1': [random.uniform(0, 100) for i in range(52)],
                'round2': [random.uniform(0, 100) for i in range(52)],
                'round3': [random.uniform(0, 100) for i in range(52)],
                'round4': [random.uniform(0, 100) for i in range(52)],
                'round5': [random.uniform(0, 100) for i in range(52)],
                'round6': [random.uniform(0, 100) for i in range(52)],
                'round7': [random.uniform(0, 100) for i in range(52)],
                'round8': [random.uniform(0, 100) for i in range(52)],
                'round9': [random.uniform(0, 100) for i in range(52)],
                'round10': [random.uniform(0, 100) for i in range(52)],
                'round11': [random.uniform(0, 100) for i in range(52)],
                'round12': [random.uniform(0, 100) for i in range(52)],
            }
        
        inputs = self.input()
        params = {}
        for k in inputs.keys():
            v = inputs[k]
            params[k] = np.expand_dims(v, 0).astype(np.float32)

        prediction = self.model(**params)

        return {
            'bids': prediction['lambda'][0, :],
            'round0': prediction['lambda_1'][0,0,:],
            'round1': prediction['lambda_1'][0,1,:],
            'round2': prediction['lambda_1'][0,2,:],
            'round3': prediction['lambda_1'][0,3,:],
            'round4': prediction['lambda_1'][0,4,:],
            'round5': prediction['lambda_1'][0,5,:],
            'round6': prediction['lambda_1'][0,6,:],
            'round7': prediction['lambda_1'][0,7,:],
            'round8': prediction['lambda_1'][0,8,:],
            'round9': prediction['lambda_1'][0,9,:],
            'round10': prediction['lambda_1'][0,10,:],
            'round11': prediction['lambda_1'][0,11,:],
            'round12': prediction['lambda_1'][0,12,:]
        }
    
    def training_data(self):

        result = self.input()

        result['chosen_bid'] = [self.chosen_bid]
        for i in range(13):
            roundname = 'round' + str(i) + '_'
            result[roundname + 'card'] = np.array([self.chosen_cards[i]])

        return result

    def store_game(self, score):
        return {
            'training': self.training_data(),
            'score': {
                'bid_result': np.array([score]),
                'rounds_result': [score] * 13,
            }
        }