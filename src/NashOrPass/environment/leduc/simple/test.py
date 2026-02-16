from src.NashOrPass.environment.leduc.simple.models.Game import Game
from src.NashOrPass.environment.leduc.simple.models.Player import Player


# Test basic game
p1 = Player(0, 100, policy="random")
p2 = Player(1, 100, policy="random")
game = Game([p1, p2], bb_amnt=1, ante=1)

try:
    game.play_hand()
    print("✓ Hand completed successfully!")
    print(f"P1 bank: {p1.bank}, P2 bank: {p2.bank}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()