import tf.train.Example
import tf.train.Feature

class PlayerState:
	def __init__(self):
		self.tricks = 0
		self.bid = -1
		self.empty_suits = [0, 0, 0, 0]

class GameState:
	def __init__(self, hand, seen, scores, tricks, bids, empty_suits):
		self.hand = [0] * 52
		self.seen = [0] * 52
		self.partner = PlayerState()
		self.me = PlayerState()
		self.left = PlayerState()
		self.right = PlayerState()
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
			"suits_empty": self.me.empty_suits + self.left.empty_suits + self.partner.empty_suits + self.right.empty_suits,
		}

	def to_example(self, label):
		features = self.to_features()
		result = {
			"label": tf.train.Feature(float_list=[label]),
		}
		for key in features.keys:
			result[key] = tf.train.Feature(float_list=features[key])

		return tf.train.Example(features=tf.train.Features(feature=features))
