"""Load trained model weights and play no_thanks against bot"""
from sys import exit
import numpy as np
import numpy.random as npr

from jax.nn import sigmoid
from jax.tree_util import tree_flatten, tree_unflatten
from jax.random import PRNGKey

import time

from game import NoThanks
from model import predict, init_random_params

# Load parameters and create leaves
npz_files = np.load("params.npz")
leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

mygame = NoThanks(4, 11)
mygame.start_game()
input_size = len(mygame.get_things())

# Get the right PyTree definition
key = PRNGKey(1)
_, temp_params = init_random_params(key, (-1, input_size))
_, treedef = tree_flatten(temp_params)

# Get parameters
params = tree_unflatten(treedef, leaves)


def print_cards_from_one_hot(one_hot_of_cards):
    return " ".join(
        str(x) for x in [i + 3 for i, x in enumerate(one_hot_of_cards) if x == 1]
    )


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
        print("-----")
        cur_player = mygame.player_turn
        print(f"Player {cur_player}")
        state = mygame.get_things()

        player_persp = mygame.get_current_player()[0]

        if cur_player == player_order:
            print(f"Center: Card {mygame.center_card} | Tokens {mygame.center_tokens}")
            print(f"Tokens: {player_persp[0]}")
            print(f"Cards {print_cards_from_one_hot(player_persp[1:])}")
            if ("t" in input()):
                game_going, rew, = mygame.take_card()
                player_store[cur_player] = (state, 0, q_vals[0])
            else:
                game_going, rew = mygame.no_thanks()
                player_store[cur_player] = (state, 1, q_vals[1])
        else:
            q_vals = predict(params, state).ravel()
            player_persp = mygame.get_current_player()[0]
            print(f"Center: Card {mygame.center_card} | Tokens {mygame.center_tokens}")
            print(f"Tokens: {player_persp[0]}")
            print(f"Cards {print_cards_from_one_hot(player_persp[1:])}")
            if sigmoid(50 * (q_vals[0] - q_vals[1])) > npr.random():
                game_going, rew, = mygame.take_card()
                player_store[cur_player] = (state, 0, q_vals[0])
            else:
                game_going, rew = mygame.no_thanks()
                player_store[cur_player] = (state, 1, q_vals[1])

        time.sleep(0.5)

    
    print(mygame.score())
    print(mygame.winning())

    for x in mygame.get_player_state_perspective():
        print(x[0])
        print(print_cards_from_one_hot(x[1:]))

