import tensorflow as tf
import max2.dataset
import threading
import time

class StreamingDataset(object):
  def __init__(self, manager, size=128):
    self._manager = manager
    self._size = size
    self._queue = []
    self._lock = threading.Lock()
    self._data_available = False
    self.data_required = True
  
  def add(self, block):
    if not self.data_required:
      return
    
    self._lock.acquire()
    if not self.data_required:
      self._lock.release()
      return
    self._queue.append(block)
    self._data_available = True
    if len(self._queue) >= self._size:
      self.data_required = False
    self._lock.release()

  def __enter__(self):
    if self._manager.generator != None:
      raise Exception("Concurrent generator running!")
    
    self._manager.generator = self

    def inner():
      while not self._data_available:
        time.sleep(0.05)
      
      self._lock.acquire()
      block = self._queue.pop()
      self.data_required = True
      if len(self._queue) == 0:
        self._data_available = False
      self._lock.release()

      yield block

    return tf.data.Dataset.from_generator(
      inner,
      output_signature = tf.TensorSpec(shape=(None, max2.dataset.size()), dtype=tf.uint8)
    )

  def __exit__(self, exception_type, exception_value, traceback):
    if self._manager.generator != self:
      raise Exception("I got deleted!")
    
    self._manager.generator = None

  