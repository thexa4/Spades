#!/usr/bin/python3


import os

from game_manager import GameManager
from braindead_player import BraindeadPlayer
from max.torch_player import TorchPlayer
from max.random_player import RandomPlayer
import numpy as np
import sys

def main():
	p1 = TorchPlayer('torchmax/results/embedding3/0002-q2.pt3')
	p2 = TorchPlayer('torchmax/results/embedding3/0002-q2.pt3')

	p1.model.caution = 0.3
	p2.model.caution = 0.3

	b_p = [p1, p2]
	t_p = [TorchPlayer('torchmax/results/try4/0038-q2.pt'), TorchPlayer('torchmax/results/try4/0038-q2.pt')]
	players = [b_p[0], t_p[0], b_p[1], t_p[1]]
	manager = GameManager(players)
	manager.should_print = True
	rounds = 10

	print("Team 0: torchmax, Team1: Random")
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
