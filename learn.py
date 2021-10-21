#!/usr/bin/python3

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import sys
import pathlib
import max2.model
import max2.dataset
import os
import random

def unpack(s):

	fields = {
		'bid_state_bids': tf.io.FixedLenFeature(shape=(4*15), dtype=tf.float32),
		'bid_state_hand': tf.io.FixedLenFeature(shape=(52), dtype=tf.float32),
		'bid_state_bags': tf.io.FixedLenFeature(shape=(2*10), dtype=tf.float32)
	}
	for i in range(13):
		roundname = 'round' + str(i) + '_'
		fields[roundname + 'seen'] = tf.io.FixedLenFeature(shape=(52), dtype=tf.float32)
		fields[roundname + 'hand'] = tf.io.FixedLenFeature(shape=(52), dtype=tf.float32)
		fields[roundname + 'played'] = tf.io.FixedLenFeature(shape=(3*52), dtype=tf.float32)
		fields[roundname + 'todo'] = tf.io.FixedLenFeature(shape=(4*26), dtype=tf.float32)
	
	fields['chosen_bid'] = tf.io.FixedLenFeature(shape=(1), dtype=tf.float32)
	for i in range(13):
		roundname = 'round' + str(i) + '_'
		fields[roundname + 'card'] = tf.io.FixedLenFeature(shape=(1), dtype=tf.float32)
	fields['bid_result'] = tf.io.FixedLenFeature(shape=(1), dtype=tf.float32)
	fields['rounds_result'] = tf.io.FixedLenFeature(shape=(13), dtype=tf.float32)

	unpacked = tf.io.parse_single_example(s, fields)
	output = {
		'bid_result': unpacked['bid_result'],
		'rounds_result': unpacked['rounds_result']
	}
	unpacked.pop('bid_result')
	unpacked.pop('rounds_result')

	return (unpacked, output)

def main():
	q = int(sys.argv[1])
	generation = int(sys.argv[2])
	if q == 0 or generation == 0:
		print('bad input')
		print(sys.argv)
		return

	folder = f'max2/data/q{q}/gen{generation:03}/samples'
	files = os.listdir(folder)
	files.remove('0001.flat.gz')
	infile = [folder + '/' + f for f in files]
	
	batchsize = 48 * 1024

	d = tf.data.Dataset.from_tensors(infile)
	d = d.unbatch()
	d = d.shuffle(1024)
	d = max2.dataset.load(d)
	d = d.shuffle(256 * 1024)
	d = d.batch(batchsize)
	d = d.prefetch(16)

	v = max2.dataset.load(folder + '/0001.flat.gz')
	v = v.batch(batchsize)
	v = v.cache()
	
	inference_model, training_model = max2.model.create()

	training_model.compile(
		loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
	)

	# 
	training_model.fit(d, validation_data=v, epochs=25)
	inference_model.save(f'max2/models/q{q}/gen{generation:03}.model')

if __name__=="__main__":
	main()
