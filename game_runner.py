#!/usr/bin/python3


import os

from game_manager import GameManager
from braindead_player import BraindeadPlayer
from max.torch_player import TorchPlayer
from max.random_player import RandomPlayer
import numpy as np
import sys

def main():
	b_p = [TorchPlayer('torchmax/results/double2/0001-q1.pt'), BraindeadPlayer()]
	t_p = [BraindeadPlayer(), BraindeadPlayer()]
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

main()
