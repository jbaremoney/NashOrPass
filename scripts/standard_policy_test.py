from src.NashOrPass.environment.leduc.simple.models.Game import Game
from src.NashOrPass.environment.leduc.simple.models.Player import Player


# Test basic game
p1 = Player(0, 100, policy="standard")
p2 = Player(1, 100, policy="random")
game = Game([p1, p2], bb_amnt=1, ante=1)

game.run_game(5)