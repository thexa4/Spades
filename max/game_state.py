import tf.train.Example
import tf.train.Feature

class player_state:
	def __init__(self):
		self.tricks = 0
		self.bid = -1
		self.empty_suits = [0, 0, 0, 0]

class game_state:
	def __init__(self, hand, seen, scores, tricks, bids, empty_suits):
		self.hand = [0] * 52
		self.seen = [0] * 52
		self.partner = player_state()
		self.me = player_state()
		self.left = player_state()
		self.right = player_state()
		self.my_score = 0
		self.opponent_score = 0

		for card in hand:
			self.hand[int(card)] = 1
		for card in seen:
			self.seen[int(card)] = 1

		self.my_score = scores[0]
		self.opponent_score = scores[1]

		self.me.tricks = tricks[0]
		self.left.tricks = tricks[1]
		self.partner.tricks = tricks[2]
		self.right.tricks = tricks[3]
		
		self.me.bid = bids[0]
		self.left.bid = bids[1]
		self.partner.bid = bids[2]
		self.right.bid = bids[3]
		
		self.me.empty_suits = empty_suits[0]
		self.left.empty_suits = empty_suits[1]
		self.partner.empty_suits = empty_suits[2]
		self.right.empty_suits = empty_suits[3]

	def to_features(self):
		return {
			"hand": self.hand,
			"seen": self.seen,
			"bids": [self.me.bid, self.left.bid, self.partner.bid, self.right.bid],
			"tricks": [self.me.tricks, self.left.tricks, self.partner.tricks, self.right.tricks],
			"scores": [self.my_score, self.opponent_score],
			"bags": [self.my_score % 10, self.opponent_score % 10],
		}

	def to_example(self, label):
		features = self.to_features()
		result = {
			#"label": tf.train.
		}
		#for key in features.keys:

