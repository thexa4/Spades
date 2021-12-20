#!/usr/bin/python3

import os
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
from multiprocessing import Pool
import serpent
from os.path import exists

def select_player(generation, models = []):

	selection_percentage = 0.3

	for model in models:
		if random.random() < selection_percentage:
			return InferencePlayer(model)
	
	return RandomPlayer()

def dataset(generation, driver, models, blocks=1, rounds=1):
	
	def play_block():
		training_player = TrainingPlayer(driver, generation)

		t_p = [select_player(generation, models), training_player]
		b_p = [select_player(generation, models) for i in range(2)]
		players = [b_p[0], t_p[0], b_p[1], t_p[1]]
		manager = GameManager(players)

		for i in range(rounds):
			manager.play_game()
		for sample in training_player.samples:
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

def work_fetcher(url):
	manager = Pyro5.api.Proxy(url)
	last_generation = None

	while True:
		unpacked = manager.fetch_todo()
		if unpacked == None:
			time.sleep(1)
			continue
		gen, q, blocksize = unpacked

		if gen != last_generation:
			last_generation = gen
			for i in range(gen - 1):
				for q in [0, 1]:
					if not exists(f'max2/models/q{q + 1}/gen{i + 1:03}.tflite'):
						os.makedirs(f'max2/models/q{q + 1}/', exist_ok=True)
						with open(f'max2/models/q{q + 1}/gen{i + 1:03}.tflite', 'xb') as f:
							f.write(serpent.tobytes(manager.get_model(i, q)))
		
		yield (gen, q, blocksize)

def perform_work(params):
	gen, q, blocksize = params

	driver = None
	models = []
	if gen > 1:
		print((2 - q, gen - 1))
		driver = max2.model.load(2 - q, gen - 1)
	
		for i in range(1, gen):
			models.append(max2.model.load(q + 1, i))
		models.reverse()

	with io.BytesIO() as b:
		with gzip.GzipFile(mode = 'wb', compresslevel = 9, fileobj = b) as f:
			for i in dataset(gen, driver, models, blocks = blocksize, rounds=1):
				arr = i.numpy()
				blockdata = arr.tobytes()
				f.write(blockdata)
		return (gen, q, b.getvalue())

def main():
	url = sys.argv[1]
	numcores = int(sys.argv[2])

	sys.excepthook = Pyro5.errors.excepthook
	manager = Pyro5.api.Proxy(url)

	iterable = work_fetcher(url)
	with Pool(numcores) as p:
		for result in p.imap_unordered(perform_work, iterable):
			gen, q, data = result

			manager.store_block(gen, q, data)
	

	
if __name__=="__main__":
	main()
