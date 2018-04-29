import copy

class ParallelPlayer
    def __init__(self, prototype, concurrent_games):
        self.instances = []
        for i in range(0, concurrent_games):
            self.instances.append(copy.deepcopy(prototype))

    def give_hand(self, game_id, cards):
        self.instances[game_id].give_hand(cards)

    def make_bid(self, game_id, bids):
        self.instances[game_id].make_bid(bids)
    
    def play_card(self, tricks):
        result = []
        for i in range(0, len(self.instances)):
            result.append(self.instances[i].play_card(tricks[i]))
        return result

    def offer_blind_nill(self, game_id, bids):
        self.instances[game_id].offer_blind_nill(bids)
    
    def receive_blind_nill_cards(self, game_id, cards):
        self.instances[game_id].receive_blind_nill_cards(cards)

    def request_blind_nill_cards(self, game_id):
        self.instances[game_id].request_blind_nill_cards()

    def announce_bids(self, game_id, bids):
        self.instances[game_id].announce_bids(bids)

    def announce_trick(self, tricks):
        for i in range(0, len(self.instances)):
            self.instances[i].announce_trick(tricks[i])

    def announce_score(self, scores):
        for i in range(0, len(self.instances)):
            self.instances[i].announce_score(scores[i])
