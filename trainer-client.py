#!/usr/bin/python3

import os

from braindead_player import BraindeadPlayer
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs Available: ", len(gpus))
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
import Pyro5.api
import time
import io
import socket
from multiprocessing import Pool
import serpent
import datetime
from os.path import exists
import Pyro5
import random

Pyro5.config.COMMTIMEOUT = 30

def select_model(generation, models = []):

	model_id = random.randrange(len(models) + 1)
	if model_id >= len(models):
		return None
	
	return models[model_id]

def dataset(generation, driver, models, blocks=1, rounds=1):
	
	def play_block():
		training_player1 = TrainingPlayer(driver, generation)
		training_player2 = TrainingPlayer(driver, generation)
		opponent_model = select_model(generation, models)

		t_p = [training_player1, training_player2]
		b_p = [RandomPlayer(), RandomPlayer()]
		if opponent_model != None:
			b_p = [InferencePlayer(opponent_model), InferencePlayer(opponent_model)]
		
		players = [b_p[0], t_p[0], b_p[1], t_p[1]]
		manager = GameManager(players)

		for i in range(rounds):
			manager.play_game()
		for sample in training_player1.samples:
			yield (sample['training'], sample['score'])
		for sample in training_player2.samples:
			yield (sample['training'], sample['score'])
	
	

	def block_dataset(i):
		fields = {
			'bid_state_bids': tf.TensorSpec(shape=(4*15), dtype=tf.float32),
			'bid_state_hand': tf.TensorSpec(shape=(52), dtype=tf.float32),
			'bid_state_bags': tf.TensorSpec(shape=(2*10), dtype=tf.float32)
		}
		for i in range(13):
			roundname = 'round' + str(i) + '_'
			fields[roundname + 'seen'] = tf.TensorSpec(shape=(52), dtype=tf.float32)
			fields[roundname + 'hand'] = tf.TensorSpec(shape=(52), dtype=tf.float32)
			fields[roundname + 'played'] = tf.TensorSpec(shape=(3*52), dtype=tf.float32)
			fields[roundname + 'todo'] = tf.TensorSpec(shape=(4*26), dtype=tf.float32)
		
		fields['chosen_bid'] = tf.TensorSpec(shape=(1), dtype=tf.float32)
		for i in range(13):
			roundname = 'round' + str(i) + '_'
			fields[roundname + 'card'] = tf.TensorSpec(shape=(1), dtype=tf.float32)
		
		return tf.data.Dataset.from_generator(lambda: play_block(),
			output_signature=(
				fields,
				{
					'bid_result': tf.TensorSpec(shape=(1), dtype=tf.float32),
					'rounds_result': tf.TensorSpec(shape=(13), dtype=tf.float32)
				},
			)
		)

	def serialize(i, o):
		features = {}
		for key in i.keys():
			features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=list(i[key])))
		for key in o.keys():
			features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=list(o[key])))

		row = tf.train.Example(features=tf.train.Features(feature=features))

		return row.SerializeToString()

	result = tf.data.Dataset.range(blocks)
	result = result.interleave(lambda x: block_dataset(x))

	result = result.batch(1024)
	result = result.map(max2.dataset.encode)
	return result.unbatch()

def work_fetcher(url, submitvars):
	state = {
		'manager': Pyro5.api.Proxy(url),
		'last_fetch': datetime.datetime.utcnow() - datetime.timedelta(minutes=60),
		'last_generation': None,
		'last_probability': None,
		'last_size': None,
	}
	manager = state['manager']
	
	def resync_models(manager, gen):
		for i in range(gen - 1):
			for q in [0, 1]:
				if not exists(f'max2/models/q{q + 1}/gen{i + 1:03}.tflite'):
					os.makedirs(f'max2/models/q{q + 1}/', exist_ok=True)
					with open(f'max2/models/q{q + 1}/gen{i + 1:03}.tflite', 'xb') as f:
						f.write(serpent.tobytes(manager.get_model(i, q)))

	def get_todo(s):
		manager = s['manager']

		if datetime.datetime.utcnow() - s['last_fetch'] > datetime.timedelta(seconds=60):
			fetched = manager.fetch_todo_params()
			new_generation = fetched[1]

			if s['last_generation'] != new_generation:
				s['last_generation'] = new_generation
				resync_models(manager, new_generation)

			if fetched[0] == 'elo':
				return fetched
			if fetched[0] == 'pause':
				return fetched

			_, _, probability, size = fetched
			s['last_fetch'] = datetime.datetime.utcnow()
			s['last_probability'] = probability
			s['last_size'] = size
		
		return ('block', s['last_generation'], random.choices([0, 1], s['last_probability'])[0], s['last_size'])


	paused_at = None
	while True:
		todo = get_todo(state)
		if todo[0] == 'pause':
			if paused_at == None:
				print('paused')
				paused_at = datetime.datetime.utcnow()
			time.sleep(1)
		else:
			if paused_at != None:
				print('resuming')
				submitvars['pausetime'] += datetime.datetime.utcnow() - paused_at
				paused_at = None
			yield todo

def perform_work(kind, params):
	if kind == 'block':
		gen, q, blocksize = params
		return perform_work_block(gen, q, blocksize)
	
	if kind == 'elo':
		gen, manager_id, teams, count = params
		return perform_work_elo(manager_id, teams, count)
	
	raise ValueError('Unknown task: ' + kind)

def perform_work_block(gen, q, blocksize):
	driver = None
	models = []
	if gen > 1:
		driver = max2.model.load(2 - q, gen - 1)
	
		for i in range(1, gen):
			models.append(max2.model.load(q + 1, i))
		models.reverse()

	count = 0
	sumtime = 0
	with io.BytesIO() as b:
		with gzip.GzipFile(mode = 'wb', compresslevel = 9, fileobj = b) as f:
			while count < blocksize:
				start = time.perf_counter()
				for i in dataset(gen, driver, models, blocks=4, rounds=1):
					count = count + 1
					arr = i.numpy()
					blockdata = arr.tobytes()
					f.write(blockdata)
				sumtime = sumtime + (time.perf_counter() - start)
		return ('block', sumtime / count, gen, q, b.getvalue())

def perform_work_elo(manager_id, teams, rounds):
	print(teams)
	def lookup_player(p):
		if p == 'random':
			return RandomPlayer()
		if p == 'braindead':
			return BraindeadPlayer()
		return InferencePlayer(max2.model.loadraw(f'max2/models/{p}'))

	team1, team2 = teams

	if len(team1) == 1:
		players = [team1[0], team2[0], team1[0], team2[0]]
	else:
		players = [team1[0], team2[0], team1[1], team2[1]]
	players = [lookup_player(x) for x in players]
	manager = GameManager(players)

	total_score = [0,0]
	wins = [0,0]
	for i in range(rounds):
		round_score = manager.play_game()
		total_score[0] = total_score[0] + int(round_score[0])
		total_score[1] = total_score[1] + int(round_score[1])

		if round_score[0] > round_score[1]:
			wins[0] = wins[0] + 1
		if round_score[1] > round_score[0]:
			wins[1] = wins[1] + 1

	return ('elo', manager_id, teams, total_score, wins)


def main():
	url = sys.argv[1]
	numcores = int(sys.argv[2])

	if exists('max2/models/q1'):
		for filename in os.listdir('max2/models/q1/'):
			if filename.endswith('.tflite'):
				os.unlink('max2/models/q1/' + filename)
	if exists('max2/models/q2'):
		for filename in os.listdir('max2/models/q2/'):
			if filename.endswith('.tflite'):
				os.unlink('max2/models/q2/' + filename)

	sys.excepthook = Pyro5.errors.excepthook
	manager = Pyro5.api.Proxy(url)
	submitvars = {
		'count': 0,
		'time': 0,
		'sumcount': 0,
		'crashed': False,
		'lastspeed': 0,
		'starttime': datetime.datetime.utcnow(),
		'pausetime': datetime.timedelta(),
		}

	iterable = work_fetcher(url, submitvars)
	with Pool(numcores, None, None, 50) as p:

		hostname = socket.gethostname()
		def handle_success(result):
			try:
				requeue(True)
				if 'manager' not in submitvars:
					submitvars['manager'] = Pyro5.api.Proxy(url)
				if result[0] == 'elo':
					_, manager_id, teams, total_score, wins = result
					submitvars['manager'].submit_elo(manager_id, teams, total_score, wins)
					return
				_, timing, gen, q, data = result

				submitvars['count'] = submitvars['count'] + 1
				submitvars['sumcount'] = submitvars['sumcount'] + 1
				submitvars['time'] = submitvars['time'] + timing

				if (submitvars['count'] % (numcores // 2)) == 0:
					perf = submitvars['time'] / submitvars['sumcount']
					submitvars['sumcount'] = 0
					submitvars['time'] = 0
					count = submitvars['count']
					submitvars['lastspeed'] = perf
					print(f'Block {count:06}: {perf:.3f} s/sample')

				submitvars['manager'].store_block(gen, q, data)
				if (submitvars['count'] % (numcores // 2)) == 0:
					submitvars['manager'].submit_client_report(hostname, submitvars['count'], submitvars['lastspeed'], numcores, submitvars['starttime'], submitvars['pausetime'].total_seconds())
			except Exception as e:
				submitvars['crashed'] = True
				print("Crash: " + e)
		
		def handle_error(error):
			submitvars['crashed'] = True
			print(error)

		def requeue(is_submit):
			if is_submit:
				try:
					if 'iterable' not in submitvars:
						submitvars['iterable'] = work_fetcher(url, submitvars)
					job = next(submitvars['iterable'])
				except Exception as e:
					submitvars['crashed'] = True
					print(e)
					return
					
			else:
				job = next(iterable)
			kind = job[0]
			params = job[1:]

			p.apply_async(perform_work, (kind, params), {}, handle_success, handle_error)

		for i in range(numcores + 2):
			requeue(False)

		while not submitvars['crashed']:
			time.sleep(1)
	

	
if __name__=="__main__":
	main()
