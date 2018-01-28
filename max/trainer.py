import tensorflow as tf
import game_state
from model import Model
from game_state import GameState

def main():
	t = Trainer("/tmp/model/")
	t.queue_sample(GameState(
		hand = [],
		seen = [],
		scores = [0, 0],
		tricks = [0, 0, 0, 0],
		bids = [3, 3, 3, 3],
		empty_suits = [[0, 0, 0, 0]] * 4,
	))
	t.set_label(10)
	t.train()

class Trainer:
	def __init__(self, model_dir):
		self.estimator = tf.estimator.Estimator(
			model_fn = Model.create_trainer,
			config = tf.estimator.RunConfig(
				model_dir = model_dir,
			),
			params = {
				'size': 12,
				'regularizer': 0.001,
				'numbers_weight': 0.01,
				'learnrate': 0.003,
			},
		)

		self.samples = {
			"hand": [],
                        "seen": [],
                        "bids": [],
                        "tricks": [],
                        "scores": [],
                        "bags": [],
                        "suits_empty": [],
		}
		self.labels = []
		self.queue = []

	def queue_sample(self, sample):
		self.queue.append(sample.to_features())

	def set_label(self, label):
		for sample in self.queue:
			for key in sample.keys():
				self.samples[key].append(sample[key])
			self.labels.append(label)
		self.queue = []

	def collect_data(self):
		label = tf.concat(self.labels, axis=-1)
		data = {}
		for key in self.samples.keys():
			data[key] = tf.concat(self.samples[key], axis = -1)

		return (data, label)
	
	def input_fn(self):
		data = (tf.data.Dataset.from_tensors(self.collect_data())
			.repeat()
			.shuffle(1024)
			.batch(32))

		return data.make_one_shot_iterator().get_next()
		

	def train(self):
		self.estimator.train(
			input_fn = self.input_fn,
			steps = 100000,
		)

		print("done")

if __name__ == "__main__":
	main()
