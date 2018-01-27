import random
import model
import tensorflow as tf

class Predictor:
	def __init__(self, model_dir):
		self.estimator = tf.estimator.Estimator(
			model_fn = model.create_model,
			config = tf.learn.RunConfig(
				model_dir = Model.create_predictor
			),
			params = {
				'size': 12,
				'regularizer': 0.001
			},
		)

	def predict(game_state):
		return random.uniform(-200, 130)
