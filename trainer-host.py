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
import trueskill
from max2.learn_sync_manager import LearnSyncManager
from max2.elo import EloManager
from max2.inference_player import InferencePlayer

def lr_schedule(epoch, lr):
	min_lr = math.log(0.00015)
	max_lr = math.log(0.0005 / math.sqrt(epoch + 1))

	length = 4
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
	
	size = 384
	#size = 4096
	depth = 20
	batchsize = 1024 * int(192 * 1024 * 1024 / size / size / depth)
	print(f"Batchsize: {batchsize}")

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
	
	inference_model, training_model = max2.model.create_v2(size)

	training_model.compile(
		loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
	)

	#tb_callback = tf.keras.callbacks.TensorBoard(f'max2/data/q{q}/gen{generation:03}/logs', update_freq=1, profile_batch=0)
	stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

	#lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
	callbacks = [stop_callback]
	#callbacks.append(stop_callback)
	training_model.fit(d, validation_data=v, epochs=30, callbacks=callbacks)
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

	trueskill.setup(draw_probability=0.2461, tau=0.005)
	elomanager_double = EloManager('double')
	elomanager_single = EloManager('single')

	daemon = Pyro5.server.Daemon(host='2001:41f0:c01:41::4252', port=51384)
	manager = LearnSyncManager(game_count = 1024 * 1024 * 48, elo_managers=[elomanager_double, elomanager_single])
	uri = daemon.register(manager, objectId='spades1')
	print(uri)
	daemon_thread = threading.Thread(target=daemon.requestLoop)
	daemon_thread.start()
	for i in range(manager.generation - 1):
		for q in [1,2]:
			path = f'max2/models/server/model-g{i+1:03}-q{q}.tflite'
			elomanager_double.add_player(lambda: InferencePlayer(max2.model.loadraw(path)), path, f'g{i+1:03}-q{q}', f'q{q}/gen{i+1:03}.tflite')
			elomanager_single.add_player(lambda: InferencePlayer(max2.model.loadraw(path)), path, f'g{i+1:03}-q{q}', f'q{q}/gen{i+1:03}.tflite')

	while True:
		if manager.is_done():
			manager.learning = True
			learn(1, manager.generation)
			learn(2, manager.generation)
			for q in [1,2]:
				path = f'max2/models/server/model-g{manager.generation:03}-q{q}.tflite'
				elomanager_single.add_player(lambda: InferencePlayer(max2.model.loadraw(path)), path, f'g{manager.generation:03}-q{q}', f'q{q}/gen{manager.generation:03}.tflite')
				elomanager_double.add_player(lambda: InferencePlayer(max2.model.loadraw(path)), path, f'g{manager.generation:03}-q{q}', f'q{q}/gen{manager.generation:03}.tflite')
			manager.advance_generation()
			manager.learning = False

			print(f'Generation {manager.generation}:')
		if random.random() > 0.5:
			if manager.generation > 1:
				print(elomanager_double.play_game())
			else:
				time.sleep(1)
		else:
			if manager.generation > 3:
				print(elomanager_single.play_game())
			else:
				time.sleep(1)

	

if __name__=="__main__":
	main()
