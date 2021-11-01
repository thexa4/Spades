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

def lr_schedule(epoch, lr):
	min_lr = math.log(0.0001)
	max_lr = math.log(0.003 / math.sqrt(epoch + 1))

	length = 10
	pos = abs((length - (epoch % (length * 2))) / length)
	interp = pos * (max_lr - min_lr) + min_lr
	return math.exp(interp)

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
	
	batchsize = 24 * 1024

	d = tf.data.Dataset.from_tensors(infile)
	d = d.unbatch()
	d = d.cache()
	d = d.shuffle(1024)
	d = max2.dataset.load(d, batchsize)
	d = d.prefetch(2)

	v = max2.dataset.load(folder + '/0001.flat.gz', batchsize)
	v = v.cache()
	
	inference_model, training_model = max2.model.create()

	training_model.compile(
		loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
	)

	tb_callback = tf.keras.callbacks.TensorBoard(f'max2/data/q{q}/gen{generation:03}/logs', update_freq=1, profile_batch=0)
	stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=False)

	lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
	callbacks = [lr_callback, tb_callback]
	#callbacks.append(stop_callback)
	training_model.fit(d, validation_data=v, epochs=50, callbacks=callbacks)
	inference_model.save(f'max2/models/q{q}/gen{generation:03}.model')

	converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
	converter.target_spec.supported_ops = [
		tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
		tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
	]
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()

	with open(f'max2/models/q{q}/gen{generation:03}.tflite', 'wb') as f:
		f.write(tflite_model)

if __name__=="__main__":
	main()
