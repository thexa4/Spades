#!/usr/bin/python3

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
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
import random

def select_player(generation, models = []):
	if random.random() < (5 / (5 + generation)) or generation <= 1 or len(models) == 0:
		return RandomPlayer()
	
	return random.choice(models)

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
	#result = result.shuffle(10240)
	#result = result.prefetch(128)
	
	for x, y in result:
		yield serialize(x,y)

def main():
	q = int(sys.argv[1])
	generation = int(sys.argv[2])
	i = int(sys.argv[3])

	if q == 0 or generation == 0 or i == 0:
		print("bad input")
		print(sys.argv)
		exit(1)
		return
	
	opponent = 3 - q
	driver = None
	if generation > 1:
		driver = max2.model.load(f'max2/models/q{opponent}/gen{(generation - 1):03}.model')
	
	models = []
	for i in range(1, generation):
		models.append(InferencePlayer(max2.model.load(f'max2/models/q{opponent}/gen{(generation - 1):03}.model')))

	pathlib.Path(f'max2/data/q{q}/gen{generation:03}/samples/').mkdir(parents=True, exist_ok=True)
	options = tf.io.TFRecordOptions(compression_type='GZIP', input_buffer_size=1024, output_buffer_size=1024,  compression_level=6)
	writer = tf.io.TFRecordWriter(f'max2/data/q{q}/gen{generation:03}/samples/{i:04}.tfrecord', options)
	for i in dataset(generation, driver, models, blocks=300, rounds=10):
		writer.write(i)
	writer.close()

	
if __name__=="__main__":
	main()
