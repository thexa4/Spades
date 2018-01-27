
class Model:

	"""
	features:
	 - hand
	 - seen
	 - bids
	 - tricks
	 - scores
	 - bags
	 - suits_empty
	"""
	def create_network(features, params):
		bids_hot = tf.concat(tf.one_hot(
			indices = features['bids'],
			depth = 15,
			dtype = tf.float(),
			name = 'bids_hot',
		), axis = -1)

		tricks_hot = tf.concat(tf.one_hot(
			indices = features['tricks'],
			depth = 14,
			dtype = tf.float(),
			name = 'tricks_hot',
		), axis = -1)

		bags_hot = tf.concat(tf.one_hot(
			indices = features['bags'],
			depth = 10,
			dtype = tf.float(),
			name = 'bags_hot',
		), axis = -1)
		

		input_layer = tf.concat([ hand, seen, bids_hot, tricks_hot, bags_hot, suits_empty ], axis = -1)

		hidden_layer = tf.layers.dense(input_layer, params.size, activation = tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizers(params.regularizer), use_bias=True)

		return tf.multiply(tf.layers.dense(hidden_layer, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(params.regularizer), use_bias=True, activation=tf.tanh), 300.0)

	def create_predictor(features, mode, params):

		output_layer = Model.create_network(features, params)
		
		predictions_dict = {
			"number": output_layer,
		}

		return tf.estimator.EstimatorSpec(
			mode = mode,
			predictions = predictions_dict,
			loss = tf.losses.get_regularization_loss(),
		)
