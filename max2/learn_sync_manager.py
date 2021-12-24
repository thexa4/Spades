import math
import random
import Pyro5.server
import threading
import serpent
import os
import datetime
from os.path import exists

@Pyro5.server.behavior(instance_mode='single')
class LearnSyncManager(object):
    def __init__(self, game_count = 8 * 1024 * 1024, blocksize = 1024):
        self.desired_block_count = math.ceil(game_count / blocksize)
        self.blocksize = blocksize

        self.generation = 1

        if exists('max2/models/curgen'):
            with open('max2/models/curgen', 'r') as f:
                self.generation = int(f.read())
        
        self._raw_set_generation(self.generation)
        self.lock = threading.Lock()
        self.hostreports = {}

    @Pyro5.server.expose
    def submit_client_report(self, hostname, samplecount, lastspeed, cores, boottime):
        self.hostreports[hostname] = {'time': datetime.datetime.utcnow(), 'count': samplecount, 'speed': lastspeed, 'cores': cores, 'start': boottime}
    
    @Pyro5.server.expose
    def get_client_reports(self):
        return self.hostreports

    @Pyro5.server.expose
    def fetch_todo(self):
        with self.lock:
            if self.blocks_left[0] + self.blocks_left[1] == 0:
                return None
            return (self.generation, random.choices([0, 1], self.blocks_left)[0], self.blocksize)
    
    @Pyro5.server.expose
    def store_block(self, gen, q, block):
        block = serpent.tobytes(block)
        with self.lock:
            if gen != self.generation:
                return
            
            block_id = self.blocks_left[q] - 1
            if block_id < 0:
                return
            self.blocks_left[q] = self.blocks_left[q] - 1

            with open(f'max2/data/q{q + 1}/gen{gen:03}/samples/{block_id:06}.flat.gz', 'xb') as f:
                f.write(block)

    @Pyro5.server.expose
    def get_generation(self):
        with self.lock:
            return self.generation

    @Pyro5.server.expose
    def get_model(self, gen, q):
        with self.lock:
            return self.models[q][gen]
    
    @Pyro5.server.expose
    def get_leaderboard(self):
        files = os.listdir('max2/models/server/')
        result = {
            'double': [],
            'single': [],
        }

        for filename in files:
            if filename.endswith('.elo'):
                mu = 25
                sigma = 8.333
                with open(f'max2/models/server/{filename}', 'r') as f:
                    mu = float(f.readline())
                    sigma = float(f.readline())
                
                label = filename.split('.')[0]
                if filename.endswith('.double.elo'):
                    result['double'].append({'label': label, 'mu': mu, 'sigma': sigma, 'trueskill': mu - 3 * sigma})
                else:
                    result['single'].append({'label': label, 'mu': mu, 'sigma': sigma, 'trueskill': mu - 3 * sigma})
        
        return result

    @Pyro5.server.expose
    def get_blocks_left(self):
        with self.lock:
            return self.blocks_left

    def is_done(self):
        with self.lock:
            return self.blocks_left[0] + self.blocks_left[1] == 0

    def _raw_set_generation(self, new_generation):
        self.generation = new_generation

        os.makedirs(f'max2/data/q1/gen{self.generation:03}/samples/', exist_ok=True)
        os.makedirs(f'max2/data/q2/gen{self.generation:03}/samples/', exist_ok=True)
        os.makedirs(f'max2/models/server/', exist_ok=True)

        self.blocks_left = [self.desired_block_count, self.desired_block_count]
        for q in [0, 1]:
            while self.blocks_left[q] > 0:
                if exists(f'max2/data/q{q + 1}/gen{self.generation:03}/samples/{self.blocks_left[q] - 1:06}.flat.gz'):
                    self.blocks_left[q] = self.blocks_left[q] - 1
                else:
                    break

        self.models = [[], []]
        for q in [0, 1]:
            for gen in range(self.generation - 1):
                with open(f'max2/models/server/model-g{gen + 1:03}-q{q + 1}.tflite', 'rb') as f:
                    self.models[q].append(bytearray(f.read()))

    def advance_generation(self):
        with self.lock:
            self._raw_set_generation(self.generation + 1)
            with open('max2/models/curgen', 'w') as f:
                f.write(str(self.generation))