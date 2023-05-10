"""Load trained model weights and play no_thanks against bot"""
from sys import exit
import numpy as np
import numpy.random as npr

import jax.random as jr
from jax.tree_util import tree_flatten, tree_unflatten

from game import NoThanks
from model import predict, init_random_params

SEED = 4
key = jr.PRNGKey(SEED)
key, sbkey = jr.split(key)

# Initialize game
mygame = NoThanks(4, 11)
mygame.start_game()
input_size = len(mygame.get_things())

# Load parameters and create leaves
npz_files = np.load("params.npz")
leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

# Get the right PyTree definition
_, temp_params = init_random_params(sbkey, (-1, input_size))
_, treedef = tree_flatten(temp_params)

# Get parameters
params = tree_unflatten(treedef, leaves)

print("player tokens", mygame.player_state[mygame.player_turn][0])
print("Center Card", mygame.center_card)
print(predict(params, mygame.get_things()).ravel())


def print_cards_from_one_hot(one_hot_of_cards):
    print(" ".join(str(x) for x in [i + 3 for i, x in enumerate(one_hot_of_cards) if x == 1]))

if __name__ == "__main__":
    player_order = npr.randint(0, 4)
    print(f"You are player number {player_order}")

    mygame = NoThanks(4, 11)
    mygame.start_game()
    game_going = 1
    experiences = []
    player_store = [
        (mygame.get_things_perspective(player), 1) for player in range(mygame.n_players)
    ]

    while game_going:
        cur_player = mygame.player_turn
        state = mygame.get_things()

        if cur_player == player_order:
            print("Your turn!")
        else:
            print(f"Player {mygame.player_turn}")
            print(f"Cards )

