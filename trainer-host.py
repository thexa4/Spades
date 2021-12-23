#!/usr/bin/python3

import os
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

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
import math
import Pyro5
import threading
import time
from max2.learn_sync_manager import LearnSyncManager
from max2.elo import EloManager
from max2.inference_player import InferencePlayer

def lr_schedule(epoch, lr):
	min_lr = math.log(0.0001)
	max_lr = math.log(0.003 / math.sqrt(epoch + 1))

	length = 10
	pos = abs((length - (epoch % (length * 2))) / length)
	interp = pos * (max_lr - min_lr) + min_lr
	return math.exp(interp)

def learn(q, generation):
	folder = f'max2/data/q{q}/gen{generation:03}/samples'
	files = os.listdir(folder)
	infile = [folder + '/' + f for f in files]
	validation_count = max(1, len(infile) // 20)

	validationsamples = infile[:validation_count]
	datasamples = infile[validation_count:]
	
	batchsize = 24 * 1024

	d = tf.data.Dataset.from_tensors(datasamples)
	d = d.unbatch()
	d = d.cache()
	d = d.shuffle(1024)
	d = max2.dataset.load(d, batchsize)
	d = d.prefetch(2)

	v = tf.data.Dataset.from_tensors(validationsamples)
	v = v.unbatch()
	v = max2.dataset.load(v, batchsize)
	v = v.cache()
	
	inference_model, training_model = max2.model.create()

	training_model.compile(
		loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
	)

	tb_callback = tf.keras.callbacks.TensorBoard(f'max2/data/q{q}/gen{generation:03}/logs', update_freq=1, profile_batch=0)
	stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
	callbacks = [lr_callback, tb_callback]
	callbacks.append(stop_callback)
	training_model.fit(d, validation_data=v, epochs=300, callbacks=callbacks)
	inference_model.save(f'max2/models/q{q}/gen{generation:03}.model')

	converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
	converter.target_spec.supported_ops = [
		tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
		tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
	]
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()

	with open(f'max2/models/server/model-g{generation:03}-q{q}.tflite', 'wb') as f:
		f.write(tflite_model)

def main():
	Pyro5.config.SERVERTYPE = 'multiplex'
	daemon = Pyro5.server.Daemon(host='2001:41f0:c01:41::4252', port=51384)
	manager = LearnSyncManager(game_count = 1024 * 1024 * 4)
	uri = daemon.register(manager, objectId='spades1')
	print(uri)
	daemon_thread = threading.Thread(target=daemon.requestLoop)
	daemon_thread.start()

	elomanager = EloManager('double')
	for i in range(manager.generation - 1):
		for q in [1,2]:
			path = f'max2/models/server/model-g{i+1:03}-q{q}.tflite'
			elomanager.add_player(lambda: InferencePlayer(max2.model.loadraw(path)), path, f'g{i+1:03}-q{q}')

	while True:
		if manager.is_done():
			learn(1, manager.generation)
			learn(2, manager.generation)
			for q in [1,2]:
				path = f'max2/models/server/model-g{manager.generation:03}-q{q}.tflite'
				elomanager.add_player(lambda: InferencePlayer(max2.model.loadraw(path)), path, f'g{manager.generation:03}-q{q}')
			manager.advance_generation()

			print(f'Generation {manager.generation}:')
		if manager.generation > 1:
			print(elomanager.play_game())
	

if __name__=="__main__":
	main()
