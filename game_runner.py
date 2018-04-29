#!/usr/bin/python3

from game_manager import GameManager
from braindead_player import BraindeadPlayer
from max.tensor_player import TensorPlayer
from max.random_player import RandomPlayer
from max.predictor import Predictor

def main():
	t_p = [TensorPlayer(Predictor("models/latest/")) for i in range(2)]
	b_p = [RandomPlayer() for i in range(2)]
	players = [b_p[0], t_p[0], b_p[1], t_p[1]]
	manager = GameManager(players)
	rounds = 100

	b_wins = 0
	t_wins = 0
	for i in range(1, rounds):
		score = manager.play_game()
		if score[0] > score[1]:
			b_wins += 1
		if score[1] > score[0]:
			t_wins += 1
		print(score)
	print(str(b_wins) + " - " + str(t_wins))


main()
