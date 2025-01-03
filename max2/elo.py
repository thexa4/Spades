from game_manager import GameManager
from braindead_player import BraindeadPlayer
from max.random_player import RandomPlayer
from max2.training_player import TrainingPlayer
import max2.model
import sys
import trueskill
import random
import itertools
import math
import concurrent.futures
import threading
from os.path import exists

class EloRoundResult:
    def __init__(self, team, score, wins):
        self.team = team
        self.score = score
        self.wins = wins

    def __str__(self):
        percentage = 'n/a%'
        if self.wins[0] + self.wins[1] > 0:
            percentage = str(int(self.wins[0] / (self.wins[0] + self.wins[1]) * 100)) + '%'
        return f'{self.team}: {self.score[0]} [{self.wins[0]}] vs {self.score[1]} [{self.wins[1]}], {percentage}'

class EloTeam:
    def __init__(self, team1, team2):
        self.teams = [team1, team2]

        players = set([*team1, *team2])
        if len(players) != len(team1) + len(team2):
            raise Exception("Cannot have duplicate players in a game")

    def __str__(self):
        if len(self.teams[0]) == 1:
            return f'[{self.teams[0][0].label} vs {self.teams[1][0].label}]'
        return f'[{self.teams[0][0].label}, {self.teams[0][1].label} vs {self.teams[1][0].label}, {self.teams[1][1].label}]'

    #https://github.com/sublee/trueskill/issues/1#issuecomment-149762508
    def win_probability(self):
        delta_mu = sum(r.score.mu for r in self.teams[0]) - sum(r.score.mu for r in self.teams[1])
        sum_sigma = sum(r.score.sigma ** 2 for r in itertools.chain(self.teams[0], self.teams[1]))
        size = len(self.teams[0]) + len(self.teams[1])
        denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
        ts = trueskill.global_env()
        return ts.cdf(delta_mu / denom)
    
    def record_score(self, team1_score, team2_score):
        rank = [team2_score, team1_score]

        t1_rank, t2_rank = trueskill.rate([[p.score for p in self.teams[0]], [p.score for p in self.teams[1]]], ranks=rank)
        players = [*self.teams[0], *self.teams[1]]
        ranks = [*t1_rank, *t2_rank]
        for player, rank in zip(players, ranks):
            player.update_rank(rank)
    
    def play(self, rounds):
        scores = [0,0]
        wins = [0,0]

        def play_round(_):
            players = []
            if len(self.teams[0]) == 1:
                players = [
                    self.teams[0][0].playerfunc(),
                    self.teams[1][0].playerfunc(),
                    self.teams[0][0].playerfunc(),
                    self.teams[1][0].playerfunc(),
                ]
            else:
                players = [
                    self.teams[0][0].playerfunc(),
                    self.teams[1][0].playerfunc(),
                    self.teams[0][1].playerfunc(),
                    self.teams[1][1].playerfunc(),
                ]
            manager = GameManager(players)
            return manager.play_game()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for score in pool.map(play_round, range(rounds)):
                scores[0] = scores[0] + score[0]
                scores[1] = scores[1] + score[1]

                if score[0] > score[1]:
                    wins[0] = wins[0] + 1
                if score[1] > score[0]:
                    wins[1] = wins[1] + 1

        return EloRoundResult(self, scores, wins)

class EloPlayer:
    def __init__(self, playerfunc, path, strategy, label, remote_path):
        self.modelpath = path
        self.elodatapath = path + '.' + strategy + '.elo'
        self.score = trueskill.Rating()
        self.label = label
        self.playerfunc = playerfunc
        self.remote_path = remote_path

        if exists(self.elodatapath):
            mu = 25
            sigma = 8.333
            with open(self.elodatapath, 'r') as f:
                mu = float(f.readline())
                sigma = float(f.readline())
            self.score = trueskill.Rating(mu = mu, sigma = sigma)
        
    def update_rank(self, rank):
        self.score = rank
        with open(self.elodatapath, 'w') as f:
            f.write(f"{rank.mu}\n")
            f.write(f"{rank.sigma}\n")

    def __lt__(self, other):
        return self.score.mu < other.score.mu


class EloManager:
    def __init__(self, strategy):
        self.lock = threading.Lock()
        self.strategy = strategy

        if strategy != 'single' and strategy != 'double':
            raise Exception("Strategy should be either single or double.")
        
        self.pool = [
            EloPlayer(lambda: BraindeadPlayer(), 'max2/models/server/braindead', self.strategy, 'Braindead', 'braindead'),
            EloPlayer(lambda: RandomPlayer(), 'max2/models/server/random', self.strategy, 'Random', 'random')
        ]
        self.lookup = {
            'braindead': self.pool[0],
            'random': self.pool[1],
        }
    
    def add_player(self, playerfunc, path, label, remote_path):
        newplayer = EloPlayer(playerfunc, path, self.strategy, label, remote_path)
        with self.lock:
            for p in self.pool:
                if p.elodatapath == newplayer.elodatapath:
                    raise Exception("Player already in pool")
            self.pool.append(newplayer)
            self.lookup[remote_path] = newplayer

    def generate_team(self):
        if self.strategy == 'double' and len(self.pool) < 2:
            raise Exception("Unable to run game with less than 4 players")
        if self.strategy == 'single' and len(self.pool) < 4:
            raise Exception("Unable to run game with less than 4 players")

        teamsize = 1
        if self.strategy == 'single':
            teamsize = 2
        
        with self.lock:
            players = random.sample(self.pool, 2 * teamsize)
        team1 = players[:teamsize]
        team2 = players[teamsize:]
        
        return EloTeam(team1, team2)

    def generate_high_entropy_team(self):
        while True:
            team = self.generate_team()
            equality = abs(team.win_probability() - 0.5) * 2
            if random.random() > equality:
                return team

    def play_game(self, team = None):
        if team == None:
            team = self.generate_high_entropy_team()

        result = team.play(10)       
        with self.lock:
            team.record_score(result.wins[0], result.wins[1])
        
        return result
