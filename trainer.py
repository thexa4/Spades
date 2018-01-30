#!/usr/bin/python3

import tensorflow as tf
import max.game_state
import random
import os
import shutil
import datetime
from max.model import Model
from max.game_state import GameState
from max.tensor_player import TensorPlayer
from max.predictor import Predictor
from max.random_player import RandomPlayer
from game_manager import GameManager
from braindead_player import BraindeadPlayer

def benchmark(path, player_type):
    t_p = [TensorPlayer(Predictor(path)) for i in range(2)]
    b_p = [player_type() for i in range(2)]
    players = [b_p[0], t_p[0], b_p[1], t_p[1]]
    manager = GameManager(players)

    b_wins = 0
    t_wins = 0
    delta_total = 0
    delta_count = 0
    for i in range(1, 10):
        score = manager.play_game()
        delta_count += 1
        delta_total += score[0]
        delta_total -= score[1]
        if score[0] > score[1]:
            b_wins += 1
        if score[1] > score[0]:
            t_wins += 1
    return (t_wins / (b_wins + t_wins), delta_total / delta_count)

def generate_moves(trainer, train_path, checkpoint_paths):
    debug_player = TensorPlayer(Predictor(train_path))
    debug_player.debug = True
    debug_player.trainer = trainer
    checkpoint_players = list(map(lambda x: TensorPlayer(Predictor(x)), checkpoint_paths))
    players = [debug_player] + checkpoint_players
    manager = GameManager(players)
    for i in range(1, 300):
        if i % 100 == 0:
            print(i)
        manager.play_game()

def main():
    train_path = "models/latest/"
    checkpoint_path = "models/checkpoints/"
    available_checkpoints = os.listdir(checkpoint_path)

    trainer = Trainer(train_path)

    while True:
        opponents = [checkpoint_path + random.choice(available_checkpoints) for x in range(0,3)]
        print("generating")
        generate_moves(trainer, train_path, opponents)

        print("training " + str(len(trainer.compiled)) + " samples")
        trainer.train()
        print("benchmarking")
        print("random: " + str(benchmark(train_path, RandomPlayer)))
        print("braindead: " + str(benchmark(train_path, BraindeadPlayer)))
        timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
        shutil.copytree(train_path, checkpoint_path + str(timestamp))
        print("-----")

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
        self.compiled = []

    def queue_sample(self, sample):
        self.queue.append(sample.to_features())

    def set_label(self, label):
        for sample in self.queue:
            self.compiled.append((sample, label))
        self.queue = []
        max_size = 2000000
        if len(self.compiled) > max_size:
            excess_elems = len(self.compiled) - max_size
            del self.compiled[:excess_elems]

    def collect_data(self):
        for e in self.compiled:
            yield e
    
    def input_fn(self):
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
            .cache()
            .repeat()
            .shuffle(10240)
            .batch(256))

        return data.make_one_shot_iterator().get_next()
        

    def train(self):
        self.estimator.train(
            input_fn = self.input_fn,
            steps = 10000,
        )

        print("done")

if __name__ == "__main__":
    main()
