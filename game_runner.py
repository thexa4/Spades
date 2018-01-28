from game_manager import GameManager
from max.tensor_player import TensorPlayer
from max.predictor import Predictor

def main():
	players = [TensorPlayer(Predictor(None)) for i in range(4)]
	manager = GameManager(players)
	print(manager.play_game())


main()
