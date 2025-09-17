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
import sqlite3

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
	if strategy == 'single':
		estimators_t1 = [estimators[pids[0]], estimators[pids[2]]]
		estimators_t2 = [estimators[pids[1]], estimators[pids[3]]]
	else:
		estimators_t1 = [estimators[pids[0]]]
		estimators_t2 = [estimators[pids[1]]]
	
	max_sigma = max([e.sigma for e in estimators_t1 + estimators_t2])
	if random.random() > (max_sigma * max_sigma) / 25:
		return list()

	players = [pool[i][1]() for i in pids]

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
	#rank = [score[0] + 500, score[1] + 500]

	t1_rank, t2_rank = trueskill.rate([estimators_t1, estimators_t2], ranks=rank)
	if strategy == 'single':
		newranks = [t1_rank[0], t2_rank[0], t1_rank[1], t2_rank[1]]
	else:
		newranks = [t1_rank[0], t2_rank[0]]

	for i in range(len(newranks)):
		estimators[pids[i]] = newranks[i]
	
	return list(zip(pids, newranks))
	

def main():

	db = sqlite3.connect("elo.db")
	db_cursor = db.cursor()
	db_cursor.execute("BEGIN TRANSACTION;")
	db_cursor.execute("CREATE TABLE IF NOT EXISTS scores (name TEXT, strategy TEXT, mu REAL, sigma REAL, UNIQUE(name, strategy));")
	db_cursor.execute("COMMIT TRANSACTION;")

	strategy = 'single'
	if len(sys.argv) >= 2:
		if sys.argv[1] == 'single':
			strategy = 'single'
		if sys.argv[1] == 'double':
			strategy = 'double'
		if sys.argv[1] == 'show':
			strategy = 'single'
			if len(sys.argv) >= 3:
				strategy = sys.argv[2]
			scores = db_cursor.execute("SELECT name, mu, sigma FROM scores WHERE strategy = ? ORDER BY (mu - 3 * sigma) DESC", [strategy]).fetchall()
			filter = ''
			if len(sys.argv) >= 4:
				filter = sys.argv[3]
			for name, mu, sigma in scores:
				if filter in name:
					print(f"{name}\t{round(mu - 3 * sigma, 2)}\t{round(mu, 2)}")
			exit(0)

	print(f'Running in {strategy} mode')
		
	with concurrent.futures.ProcessPoolExecutor(max_workers=min(61, os.cpu_count() // 4)) as executor:
		count = 0
		while True:
			pool = [
				('Braindead', BraindeadPlayer),
				('Random', RandomPlayer),
			]
				
			for experiment in os.listdir('torchmax/results'):
				if os.path.isdir(f"torchmax/results/{experiment}"):
					for path in os.listdir(f"torchmax/results/{experiment}"):
						if 'q1_ckpt.pt' in path or 'q2_ckpt.pt' in path:
							continue
						if path.endswith('.pt') or path.endswith('.pt2') or path.endswith('.pt3'):
							fullpath = f"torchmax/results/{experiment}/{path}"
							pool.append((experiment + '-' + path[:-3], functools.partial(TorchPlayer, fullpath)))

			if strategy == 'double':
				if len(pool) % 2 == 1:
					pool.append(('Random', RandomPlayer))
			if strategy == 'single':
				for _ in range(4 - ((len(pool) - 1) % 4)):
					pool.append(('Random', RandomPlayer))

			estimators = [trueskill.Rating() for p in pool]
			for pid in range(len(pool)):
				db_lookup = db_cursor.execute("SELECT mu, sigma FROM scores WHERE name = ? AND strategy = ?;", [pool[pid][0], strategy]).fetchone()
				if db_lookup != None:
					estimators[pid] = trueskill.Rating(db_lookup[0], db_lookup[1])


			print("Round " + str(count))
			leaderboard = list(range(len(pool)))
			leaderboard.sort(key=lambda i: estimators[i].mu - 3 * estimators[i].sigma + random.random() * 5, reverse=True)
			for i in leaderboard:
				print(pool[i][0].ljust(25) + str(estimators[i]))
			print()

			if strategy == 'double':
				if (count % 10) < 9:
					pairs = list(zip(leaderboard[::2], leaderboard[1::2]))
				else:
					shuffled_leaderboard = leaderboard.copy()
					random.shuffle(shuffled_leaderboard)
					pairs = list(zip(shuffled_leaderboard[::2], shuffled_leaderboard[1::2]))
			elif strategy == 'single':
				if (count % 10) < 9:
					pairs = list(zip(leaderboard[::4], leaderboard[1::4], leaderboard[2::4], leaderboard[3::4]))
				else:
					shuffled_leaderboard = leaderboard.copy()
					random.shuffle(shuffled_leaderboard)
					pairs = list(zip(shuffled_leaderboard[::4], shuffled_leaderboard[1::4], shuffled_leaderboard[2::4], shuffled_leaderboard[3::4]))
			else:
				crash()
			
			count = count + 1

			queries = []
			for updates in executor.map(play_game, [(pool, estimators, pair) for pair in pairs]):
				for pid, newestimator in updates:
					estimators[pid] = newestimator
					queries.append([pool[pid][0], strategy, newestimator.mu, newestimator.sigma, newestimator.mu, newestimator.sigma])

			db_cursor.execute("BEGIN TRANSACTION;")
			for update in queries:
				db_cursor.execute("INSERT INTO scores(name, strategy, mu, sigma) VALUES(?, ?, ?, ?) ON CONFLICT(name, strategy) DO UPDATE SET mu = ?, sigma = ?;", update)
			db_cursor.execute("COMMIT TRANSACTION;")


if __name__ == '__main__':
	freeze_support()
	main()
