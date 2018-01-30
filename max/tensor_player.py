"""Spades IA"""
import random
from max.base_player import BasePlayer
from max.game_state import GameState

class TensorPlayer(BasePlayer):
    """"Tensor trained player """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.debug = False
        self.trainer = False
        self.score = 0
        self.opponent_score = 0

    def make_bid(self, bids):
        """
        Ask this player to make a bid at the start of a round

        bids: a dict containing all bids so far, with keys 0-3 (player_id) and values 0-13 or "B" for Blind Nill
        return value: An integer between 0 (a Nill bid) and 13 minus the teammate's bid (inclusive)
        """
        return random.randint(0, 6)

    def get_expected_point_delta(self, state):
        """Get expected point delta from tenser flow AI."""
        return self.predictor.predict(state)

    def play_card(self, trick, valid_cards):
        """
        Ask this player to play a card, given the trick so far

        trick: a Trick object with the cards played so far
        return value: a Card object present in your hand
        """
        self.seen.update(trick.cards.values())
                
        weights = []
        cards = []
        states = {}
        for card in valid_cards:
            hand = filter(lambda c: c != card, self.hand)
            state = GameState(hand=hand,            
                    seen=self.seen, 
                    scores=[self.score,self.opponent_score],
                    tricks=self.tricksWon, 
                    bids=self.bids, 
                    empty_suits=self.empty_suits)
            states[card] = state
            delta = self.get_expected_point_delta(state)
            weights.append(delta)
            cards.append(card)

        selection = cards[TensorPlayer.weighted_choice_sub(weights)]
        if self.debug:
            self.trainer.queue_sample(states[selection])
        return selection
 
    def weighted_choice_sub(weights):
        offset = 9999999
        maxval = -9999999
        for w in weights:
                if w < offset:
                        offset = w
                if w > maxval:
                        maxval = w
        
        scaled = list(map(lambda x: (x - offset) / (maxval - offset + 0.00001) + 0.01, weights))
        rnd = random.random() * sum(scaled)
        for i, w in enumerate(scaled):
                
            rnd -= w
            if rnd <= 0:
                return i

        print(rnd)
        print(weights)
        print(scaled)
      
    def announce_score(self, score):
        prevdelta = self.score - self.opponent_score
        self.score = score[0]
        self.opponent_score = score[1]
        newdelta = self.score - self.opponent_score

        if self.debug:
            self.trainer.set_label(newdelta - prevdelta)
