from models.bidder import Bidder
from models.braindead import BrainDead
from models.fullplayer3 import FullPlayer3

from simulator import run_game

path = 'results/embedding5/q1_ckpt.pt3'
target = FullPlayer3.load(path, 0, 'cpu')[0]
target.debug = True
target.eval()
double = FullPlayer3.load(path, 0, 'cpu')[0]

models = [
    target,
    BrainDead(),
    double,
    BrainDead(),
]

run_game(2, models, should_print=True)
