import random
import os
from max.model import Model
from max.game_state import GameState
import tensorflow as tf

class Predictor:
    def __init__(self, model_dir):
        if not os.path.isdir(model_dir):
            train_estimator = tf.estimator.Estimator(
                model_fn = Model.create_trainer,
                config = tf.estimator.RunConfig(
                    model_dir = model_dir,
                ),
                params = {
                    'regularizer': 0.001,
                    'numbers_weight': 0.01,
                    'win_weight': 0.1,
                    'learnrate': 0.003,
                },
            )
            self.samples = [(GameState(
                hand = [],
                seen = [],
                scores = [0, 0],
                tricks = [0, 0, 0, 0],
                bids = [3, 3, 3, 3],
                empty_suits = [[0, 0, 0, 0]] * 4,
            ).to_features(), { "score_delta": 0, "win_chance": 0, })]
            train_estimator.train(
                self.train_input_fn,
                steps = 1,
            )

        self.estimator = tf.estimator.Estimator(
            model_fn = Model.create_predictor,
            config = tf.estimator.RunConfig(
                model_dir = model_dir
            ),
            params = {
                'regularizer': 0.001
            },
        )
    
    def collect_data(self):
        for sample in self.samples:
            yield sample
    
    def train_input_fn(self):
        datadesc = ({
            "hand": tf.int32,
            "seen": tf.int32,
            "bids": tf.int32,
            "tricks": tf.int32,
            "scores": tf.int32,
            "bags": tf.int32,
            "suits_empty": tf.int32,
        }, {
            "score_delta": tf.int32,
            "win_chance": tf.int32,
        })
        data = (tf.data.Dataset.from_generator(self.collect_data, datadesc)
            .batch(1))

        return data.make_one_shot_iterator().get_next()
    
    def input_fn(self):
        datadesc = {
            "hand": tf.int32,
            "seen": tf.int32,
            "bids": tf.int32,
            "tricks": tf.int32,
            "scores": tf.int32,
            "bags": tf.int32,
            "suits_empty": tf.int32,
        }
        data = (tf.data.Dataset.from_generator(self.collect_data, datadesc)
            .batch(100))

        return data.make_one_shot_iterator().get_next()

    def predict_raw(self, game_states):
        self.samples = map(lambda x: x.to_features(), game_states)
        return self.estimator.predict(self.input_fn)

    def predict(self, game_states):
        return list(map(lambda x: x["number"], self.predict_raw(game_states)))
    
    def predict_win(self, game_states):
        return list(map(lambda x: x["win"], self.predict_raw(game_states)))
