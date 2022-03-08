#!/usr/bin/python3

import Pyro5.api
import sys

def display_leaderboard(strategy, board, limit):
	if limit > 1000:
		print(f"{strategy + ' all':<16s}\tTskill\tMu")
	else:
		print(f"{strategy + ' first ' + str(limit):<16s}\tTskill\tMu")
	ranked = sorted(board, key = lambda x: x['trueskill'], reverse=True)
	for rank in ranked[:limit]:
		label = rank['label']
		parts = label.split('-')
		if len(parts) == 3:
			label = '-'.join([parts[0], parts[2], parts[1]])
		print(f"{label:<16s}\t{rank['trueskill']:.2f}\t{rank['mu']:.2f}")
	print()

def main():
	url = sys.argv[1]
	limit = sys.maxsize
	strategy = 'all'
	if len(sys.argv) > 2:
		limit = int(sys.argv[2])
	if len(sys.argv) > 3:
		strategy = sys.argv[3]

	sys.excepthook = Pyro5.errors.excepthook
	manager = Pyro5.api.Proxy(url)

	leaderboard = manager.get_leaderboard()
	if strategy != 'single':
		display_leaderboard('Double', leaderboard['double'], limit)
	if strategy != 'double':
		display_leaderboard('Single', leaderboard['single'], limit)
	
if __name__=="__main__":
	main()
