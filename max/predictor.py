import random
from max.model import Model
import tensorflow as tf

class Predictor:
	def __init__(self, model_dir):
		self.estimator = tf.estimator.Estimator(
			model_fn = Model.create_predictor,
			config = tf.estimator.RunConfig(
				model_dir = model_dir
			),
			params = {
				'size': 64,
				'regularizer': 0.001
			},
		)

	def predict(self, game_state):
		return random.uniform(-200, 130)
