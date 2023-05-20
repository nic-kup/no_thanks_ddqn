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



def print_cards_from_one_hot(one_hot_of_cards):
    return " ".join(
        str(x) for x in [i + 3 for i, x in enumerate(one_hot_of_cards) if x == 1]
    )


if __name__ == "__main__":
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
    player_order = npr.randint(0, 4)

    print(f"You are player number {player_order}")

    mygame = NoThanks(4, 11)
    mygame.start_game()
    game_going = 1
    experiences = []

    time.sleep(0.5)
    print("|-|-|-|-|")

    while game_going:
        cur_player = mygame.player_turn
        print(f"Player {cur_player}" + ", Your Turn!" * (cur_player == player_order))
        state = mygame.get_things()

        player_persp = mygame.get_current_player()[0]

        if cur_player == player_order:
            print(f"Tokens {player_persp[0]:<2} Cards {print_cards_from_one_hot(player_persp[1:])}")
            print(f"Center Tokens {mygame.center_tokens} Card {mygame.center_card}")
            if "t" in input():
                game_going, rew = mygame.take_card()
            else:
                game_going, rew = mygame.no_thanks()
        else:
            q_vals = predict(params, state).ravel()
            player_persp = mygame.get_current_player()[0]
            print(f"Tokens {player_persp[0]:<2} Cards {print_cards_from_one_hot(player_persp[1:])}")
            print(f"Center Tokens {mygame.center_tokens} Card {mygame.center_card}")
            if sigmoid(50 * (q_vals[0] - q_vals[1])) > npr.random():
                print("Take!")
                game_going, rew = mygame.take_card()
            else:
                print("No Thanks!")
                game_going, rew = mygame.no_thanks()

        time.sleep(0.5)
        print("-----")

    print(mygame.score())
    print(mygame.winning())

    for x in mygame.get_player_state_perspective():
        print(f"{x[0]:<3}|{print_cards_from_one_hot(x[1:])}")
