import hashlib
import math
import random
import Pyro5.server
import threading
import serpent
import os
import datetime
import gzip
from torchmax.streaming_dataset import StreamingDataset
from os.path import exists
import numpy as np
import torchmax.dataset
from torchmax.elo import EloRoundResult, EloTeam

@Pyro5.server.behavior(instance_mode='single')
class LearnSyncManager(object):
    def __init__(self, blocksize = 1024, elo_managers = [], elosize = 10):
        self.blocksize = blocksize
        self.elo_managers = elo_managers
        self.elosize = elosize
        self.elopercentage = 0.03
        self.learning = False

        self.generation = 1

        self.generator = None
        self.current_q = None

        if exists('max2/models/curgen'):
            with open('max2/models/curgen', 'r') as f:
                self.generation = int(f.read())
        
        self._raw_set_generation(self.generation)
        self.lock = threading.Lock()
        self.hostreports = {}

    def create_dataset(self, q):
        self._current_q = q
        return StreamingDataset(self)

    @Pyro5.server.expose
    def submit_client_report(self, hostname, samplecount, lastspeed, cores, boottime, pausetime):
        self.hostreports[hostname] = {'time': datetime.datetime.utcnow(), 'count': samplecount, 'speed': lastspeed, 'cores': cores, 'start': boottime, 'pause': pausetime}
    
    @Pyro5.server.expose
    def ping(self):
        pass

    @Pyro5.server.expose
    def get_elo_percentage(self):
        if self.learning:
            return 1.0
        return self.elopercentage

    @Pyro5.server.expose
    def get_client_reports(self):
        return self.hostreports

    @Pyro5.server.expose
    def create_elo_todo(self):
        if len(self.elo_managers) == 0:
            return ('pause', self.generation)
        
        manager_id = random.choice([0,1])
        manager = self.elo_managers[manager_id]

        team = manager.generate_high_entropy_team()
        team1 = [x.remote_path for x in team.teams[0]]
        team2 = [x.remote_path for x in team.teams[1]]
        return ('elo', self.generation, manager_id, [team1, team2], self.elosize)

    @Pyro5.server.expose
    def fetch_todo(self):
        with self.lock:
            if self.get_blocks_left()[0] + self.get_blocks_left()[1] == 0:
                return self.create_elo_todo()
            if random.random() > (1 - self.elopercentage):
                return self.create_elo_todo()
            return ('block', self.generation, random.choices([0, 1], self.get_blocks_left())[0], self.blocksize)

    @Pyro5.server.expose
    def fetch_todo_params(self):
        with self.lock:
            todo = self.get_blocks_left()[0] + self.get_blocks_left()[1]
            if todo == 0:
                return self.create_elo_todo()
            return ('block', self.generation, self.get_blocks_left(), self.blocksize)
    
    @Pyro5.server.expose
    def store_block(self, gen, q, block):
        if q != self._current_q - 1:
            return
        
        if gen != self.generation:
            print("wrong gen")
            return
            
        if self.generator == None:
            print("no gen")
            return
        
        block = serpent.tobytes(block)
        decompressed = gzip.decompress(block)
        decompressed = np.frombuffer(decompressed, dtype=np.byte)
        decompressed = np.reshape(decompressed, (-1, torchmax.dataset.size()))

        with self.lock:
            if q != self._current_q - 1:
                print("wrong q")
                return
            if gen != self.generation:
                print("wrong gen")
                return

            if self.generator != None:
                self.generator.add(decompressed)
    
    @Pyro5.server.expose
    def submit_elo(self, manager_id, teams, total_score, wins):
        manager = self.elo_managers[manager_id]

        team1, team2 = teams
        team1 = [manager.lookup[x] for x in team1]
        team2 = [manager.lookup[x] for x in team2]

        team = EloTeam(team1, team2)
        result = EloRoundResult(team, total_score, wins)
        with manager.lock:
            team.record_score(result.wins[0], result.wins[1])
        if not self.learning:
            print(f'{result} [foreign]')

    @Pyro5.server.expose
    def get_generation(self):
        with self.lock:
            return self.generation

    @Pyro5.server.expose
    def get_model_digest(self, gen, q):
        with self.lock:
            return hashlib.sha3_256(self.models[q][gen]).hexdigest()

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
        try:
            if self.generator != None and not self.generator.data_required:
                return [0, 0]
        except Exception:
            pass
        if self._current_q == 1:
            return [1024, 0]
        if self._current_q == 2:
            return [0, 1024]
        return [0, 0]

    def _raw_set_generation(self, new_generation):
        self.generator = None
        self.generation = new_generation

        os.makedirs(f'max2/models/server/', exist_ok=True)

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