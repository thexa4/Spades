#!/usr/bin/python3

import os
import socketserver

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import sys
import pathlib
import max2.model

class PredictionServer(socketserver.ThreadingTCPServer):
  def __init__(self, server_address, RequestHandlerClass, arg1, arg2):
    SocketServer.ThreadingTCPServer.__init__(self, 
                                              server_address, 
                                              RequestHandlerClass):
    self.lock = 

class PredictionServerHandler(socketserver.StreamRequestHandler):
  def setup(self):
    self.

  def handle(self):
    self.data = self.rfile.readline().strip()
    self.wfile.write(self.data.upper())

def main():
	q = int(sys.argv[1])
	generation = int(sys.argv[2])

	if q == 0 or generation == 0
		print("bad input")
		print(sys.argv)
		exit(1)
		return
	
  model = max2.model.load(q, generation)



if __name__ == "__main__":
    main()