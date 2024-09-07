from models.bidder import Bidder
from models.braindead import BrainDead
from models.fullplayer import FullPlayer

from simulator import run_game

models = [
    #Bidder.load('results/q1-0000.pt', 0),
    FullPlayer(0),
    BrainDead(),
    BrainDead(),
    BrainDead(),
]

run_game(2, models, should_print=True)
