#!/usr/bin/python3


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
import trueskill
import random
import itertools
import math

#https://github.com/sublee/trueskill/issues/1#issuecomment-149762508
def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)

def main():

	strategy = 'single'
	if len(sys.argv) >= 2:
		if sys.argv[1] == 'single':
			strategy = 'single'
		if sys.argv[1] == 'double':
			strategy = 'double'
	print(f'Running in {strategy} mode')
	
	pool = [
		('Braindead', lambda: BraindeadPlayer()),
		('Random', lambda: RandomPlayer()),
	]

	q1_models = []#[(f'Max {i:03}A', max2.model.load(1, i)) for i in range(1, 4)]
	q2_models = [(f'Max {i:03}B', max2.model.load(2, i)) for i in range(1, 7)]
	#q3_models = [('Inference 3 gen ' + str(i), max2.model.load(3, i)) for i in range(1,2)]#, 5)]
	#q4_models = []#[('Inference 4 gen ' + str(i), max2.model.load(4, i)) for i in range(1, 6)]

	models = q1_models + q2_models# + q3_models + q4_models

	for pair in models:
		name, model = pair
		pool.append((name, lambda: InferencePlayer(model)))

	estimators = [trueskill.Rating() for p in pool]

	count = 0
	for i in range(5000):
		print("Round " + str(count))
		leaderboard = list(range(len(pool)))
		leaderboard.sort(key=lambda i: estimators[i].mu, reverse=True)
		for i in leaderboard:
			print(pool[i][0].ljust(25) + str(estimators[i]))
		count = count + 1
		print()

		pids = random.sample(range(len(pool)), 4)
		if strategy == 'double':
			teamids = random.sample(range(len(pool)), 2)
			pids = list(teamids) + list(teamids)
		
		t1_desc = pool[pids[0]][0] + ", " + pool[pids[2]][0]
		print(t1_desc + " vs " + pool[pids[1]][0] + ", " + pool[pids[3]][0])
		players = [pool[i][1]() for i in pids]
		if strategy == 'single':
			estimators_t1 = [estimators[pids[0]], estimators[pids[2]]]
			estimators_t2 = [estimators[pids[1]], estimators[pids[3]]]
		else:
			estimators_t1 = [estimators[pids[0]]]
			estimators_t2 = [estimators[pids[1]]]
	
		winpercent = win_probability(estimators_t1, estimators_t2)
		t1_winpercent = '{:.1%}'.format(winpercent)
		print(t1_winpercent.rjust(len(t1_desc)) + ' vs {:.1%}'.format(1-winpercent))

		manager = GameManager(players)
		score = manager.play_game()

		print(str(score[0]).rjust(len(t1_desc)) + ' vs ' + str(score[1]))

		rank = [0, 0]
		if score[0] < score[1]:
			rank[0] = 1
		if score[1] < score[0]:
			rank[1] = 1

		t1_rank, t2_rank = trueskill.rate([estimators_t1, estimators_t2], ranks=rank)
		if strategy == 'single':
			newranks = [t1_rank[0], t2_rank[0], t1_rank[1], t2_rank[1]]
		else:
			newranks = [t1_rank[0], t2_rank[0]]

		for i in range(len(newranks)):
			estimators[pids[i]] = newranks[i]
		
		#input('Press enter to continue: ')

	quit()
	model = max2.model.load(2,5)
	t_p = [InferencePlayer(model), InferencePlayer(model)]
	#b_p = [BraindeadPlayer() for i in range(2)]
	model_prev = max2.model.load(2,4)
	b_p = [InferencePlayer(model_prev), InferencePlayer(model_prev)]
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
