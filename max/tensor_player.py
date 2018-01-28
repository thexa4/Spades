"""Spades IA"""

class TensorPlayer(BasePlayer):
    """"Tensor trained player """

    def __init__(self, predictor):
        super().__init__(self)
        self.predictor = predictor

    def make_bid(self, bids):
        """
        Ask this player to make a bid at the start of a round

        bids: a dict containing all bids so far, with keys 0-3 (player_id) and values 0-13 or "B" for Blind Nill
        return value: An integer between 0 (a Nill bid) and 13 minus the teammate's bid (inclusive)
        """
        return randint(0, 13)

    def get_expected_point_delta(self, state):
        """Get expected point delta from tenser flow AI."""
        return self.predictor.predict(state)

    def play_card(self, trick, valid_cards):
        """
        Ask this player to play a card, given the trick so far

        trick: a Trick object with the cards played so far
        return value: a Card object present in your hand
        """
        self.seen.update(trick.cards)
                
        weights = []
        cards = []
        for card in valid_cards:
            hand = filter(lambda c: c != card, self.hand)
            state = GameState(hand=hand,            
                    seen=self.seen, 
                    scores=[self.score,self.opponent_score],
                    tricks=[self.tricks], 
                    bids=self.bids, 
                    empty_suits=[self.empty_suits])
            delta = get_expected_point_delta(state)
            weights.append(delta)
            cards.append(card)

        return cards[get_weighted_random(weights)]
 
    def weighted_choice_sub(weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i
      
