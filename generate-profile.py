#!/usr/bin/python3

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from game_manager import GameManager
from max.random_player import RandomPlayer
from max2.training_player import TrainingPlayer
from max2.inference_player import InferencePlayer
import numpy as np
import sys
import concurrent.futures
import pathlib
import max2.model
import max2.dataset
import random
import gzip

def select_player(generation, models = []):
    return InferencePlayer(random.choice(models))

def play_block(generation, driver, models):
    training_player = TrainingPlayer(driver, generation)

    t_p = [select_player(generation, models), training_player]
    b_p = [select_player(generation, models) for i in range(2)]
    players = [b_p[0], t_p[0], b_p[1], t_p[1]]
    manager = GameManager(players)

    for i in range(10):
        manager.play_game()
        print("round")

def main():
    q = 1
    generation = 2
    
    opponent = 3 - q
    driver = None
    if generation > 1:
        driver = max2.model.load(opponent, generation - 1)
    
    models = []
    for i in range(1, generation):
        model = max2.model.load(q, i)
        models.append(model)

    play_block(generation, driver, models)
    

    
if __name__=="__main__":
    main()
