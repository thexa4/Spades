#!/usr/bin/python3


from multiprocessing import freeze_support
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from game_manager import GameManager
from braindead_player import BraindeadPlayer
from max.random_player import RandomPlayer
from max.torch_player import TorchPlayer
import functools
import numpy as np
import sys
import trueskill
import random
import itertools
import math
import concurrent.futures

#https://github.com/sublee/trueskill/issues/1#issuecomment-149762508
def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)

def play_game(args):
	pool, estimators, pids = args

	strategy = 'single'
	if len(pids) == 2:
		strategy = 'double'
		pids = list(pids) + list(pids)

	t1_desc = pool[pids[0]][0] + ", " + pool[pids[2]][0]
	result = t1_desc + " vs " + pool[pids[1]][0] + ", " + pool[pids[3]][0]
	players = [pool[i][1]() for i in pids]
	if strategy == 'single':
		estimators_t1 = [estimators[pids[0]], estimators[pids[2]]]
		estimators_t2 = [estimators[pids[1]], estimators[pids[3]]]
	else:
		estimators_t1 = [estimators[pids[0]]]
		estimators_t2 = [estimators[pids[1]]]

	winpercent = win_probability(estimators_t1, estimators_t2)
	t1_winpercent = '{:.1%}'.format(winpercent)
	result = result + "\n"
	result = result + t1_winpercent.rjust(len(t1_desc)) + ' vs {:.1%}'.format(1-winpercent)

	manager = GameManager(players)
	score = manager.play_game()

	result = result + "\n"
	result = result + str(score[0]).rjust(len(t1_desc)) + ' vs ' + str(score[1])
	print(result)

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
	
	return list(zip(pids, newranks))
	

def main():

	strategy = 'single'
	if len(sys.argv) >= 2:
		if sys.argv[1] == 'single':
			strategy = 'single'
		if sys.argv[1] == 'double':
			strategy = 'double'
	print(f'Running in {strategy} mode')
	
	pool = [
		('Braindead', BraindeadPlayer),
		('Random', RandomPlayer),
	]
	for path in os.listdir('torchmax/results-try1'):
		if path.endswith('.pt'):
			fullpath = 'torchmax/results-try1/' + path
			pool.append(('try1-' + path[:-3], functools.partial(TorchPlayer, fullpath)))
	for path in os.listdir('torchmax/results'):
		if path.endswith('.pt'):
			fullpath = 'torchmax/results/' + path
			pool.append(('try2-' + path[:-3], functools.partial(TorchPlayer, fullpath)))

	if strategy == 'double':
		if len(pool) % 2 == 1:
			pool.append(('Random', RandomPlayer))
	if strategy == 'single':
		for _ in range(4 - ((len(pool) - 1) % 4)):
			pool.append(('Random', RandomPlayer))

	estimators = [trueskill.Rating() for p in pool]

	with concurrent.futures.ProcessPoolExecutor(max_workers=min(61, os.cpu_count() * 2)) as executor:

		count = 0
		for i in range(1000):
			print("Round " + str(count))
			leaderboard = list(range(len(pool)))
			leaderboard.sort(key=lambda i: estimators[i].mu, reverse=True)
			for i in leaderboard:
				print(pool[i][0].ljust(25) + str(estimators[i]))
			print()

			if (count % 6) == 0:
				pairs = list(zip(leaderboard[::2], leaderboard[1::2]))
			elif (count % 6) == 3:
				pairs = [(leaderboard[0], leaderboard[-1])] + list(zip(leaderboard[1::2], leaderboard[2:-1:2]))
			else:
				shuffled_leaderboard = leaderboard.copy()
				random.shuffle(shuffled_leaderboard)
				pairs = list(zip(shuffled_leaderboard[::2], shuffled_leaderboard[1::2]))
			
			count = count + 1

			if strategy == 'single':
				crash()

			for updates in executor.map(play_game, [(pool, estimators, pair) for pair in pairs]):
				for pid, newestimator in updates:
					estimators[pid] = newestimator
			
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

if __name__ == '__main__':
	freeze_support()
	main()
