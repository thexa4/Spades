#!/usr/bin/python3

from game_manager import GameManager
from braindead_player import BraindeadPlayer
from max.tensor_player import TensorPlayer
from max.random_player import RandomPlayer
from max.predictor import Predictor
from max2.inference_player import InferencePlayer
from max2.training_player import TrainingPlayer
import numpy as np
import max2.model
import sys

def main():
	model = max2.model.load(1, 2)
	t_p = [InferencePlayer(model), TrainingPlayer(model)]
	#b_p = [BraindeadPlayer() for i in range(2)]
	model_prev = max2.model.load(1,1)
	b_p = [InferencePlayer(model_prev), TrainingPlayer(model_prev)]
	players = [b_p[0], t_p[0], b_p[1], t_p[1]]
	manager = GameManager(players)
	rounds = 10

	b_wins = 0
	t_wins = 0
	for i in range(rounds):
		score = manager.play_game()
		if score[0] > score[1]:
			b_wins += 1
		if score[1] > score[0]:
			t_wins += 1
		print(score)
	print(str(b_wins) + " - " + str(t_wins))

	print(len(t_p[1].samples))

main()
