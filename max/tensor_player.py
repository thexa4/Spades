"""Spades IA"""

class TensorPlayer(BasePlayer):
    """"Tensor trained player """

    def make_bid(self, bids):
        """
        Ask this player to make a bid at the start of a round

        bids: a dict containing all bids so far, with keys 0-3 (player_id) and values 0-13 or "B" for Blind Nill
        return value: An integer between 0 (a Nill bid) and 13 minus the teammate's bid (inclusive)
        """
        return 3

    def get_expected_point_delta(self, state):
        """Get expected point delta from tenser flow AI."""
        return 4 

    def play_card(self, trick, valid_cards):
        """
        Ask this player to play a card, given the trick so far

        trick: a Trick object with the cards played so far
        return value: a Card object present in your hand
        """
        self.seen.update(trick.cards)

        
        deltas = {}
        for card in valid_cards:

            state = GameState(hand=self.hand, 
                    seen=self.seen, 
                    scores=[self.score,self.opponent_score],
                    tricks=[self.tricks], 
                    bids=self.bids, 
                    empty_suits=[self.empty_suits])
            delta = get_expected_point_delta(state)
            deltas[card] = delta      

        return get_weighted_random(deltas)

    def get_weighted_random(cards):
        return cards.keys().pop() # TODO weighted random please.

