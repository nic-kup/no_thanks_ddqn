"""Load trained model weights and play no_thanks against bot"""
from sys import exit
import numpy as np
import numpy.random as npr

from jax import jit

from jax.nn import sigmoid
from jax.tree_util import tree_flatten, tree_unflatten
from jax.random import PRNGKey

import time

import matplotlib.pyplot as plt

from game import NoThanks
from model import predict, init_random_params


def print_cards_from_one_hot(one_hot_of_cards):
    return " ".join(
        str(x) for x in [i + 3 for i, x in enumerate(one_hot_of_cards) if x == 1]
    )


if __name__ == "__main__":

    @jit
    def pred_q_values(params, state):
        return predict(params, state)[0]

    # Load parameters and create leaves
    npz_files = np.load("all_params.npz")
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
    print("Do you want to play?")
    if "n" in input():
        player_order = -1

    print(f"You are player number {player_order}")

    mygame = NoThanks(4, 11)
    mygame.start_game()
    game_going = 1
    game_states = [[] for i in range(4)]
    time_states = [[] for i in range(4)]

    time.sleep(0.5)
    print("|-|-|-|-|")

    turn_num = 0
    while game_going:
        cur_player = mygame.player_turn
        print(f"Player {cur_player}" + ", Your Turn!" * (cur_player == player_order))
        state = mygame.get_things().reshape((1, -1))
        game_states[cur_player].append(state.squeeze())
        time_states[cur_player].append(turn_num)

        turn_num += 1

        player_cards = mygame.player_cards[cur_player]
        player_tkns = mygame.get_player_tokens_int(cur_player)
        q_vals = pred_q_values(params, state).ravel()

        if cur_player == player_order:
            print(
                f"Tokens {player_tkns:<2} Cards {print_cards_from_one_hot(player_cards)}"
            )
            print(f"Center Tokens {mygame.center_tokens} Card {mygame.center_card}")
            if "t" in input():
                game_going, rew = mygame.take_card()
            else:
                game_going, rew = mygame.no_thanks()
        else:
            player_persp = mygame.get_current_player()[0]
            print(
                f"Tokens {player_tkns:<2} Cards {print_cards_from_one_hot(player_cards)}"
            )
            print(f"Center Tokens {mygame.center_tokens} Card {mygame.center_card}")
            if q_vals[0] > q_vals[1]:
                print("Take!")
                game_going, rew = mygame.take_card()
            else:
                print("No Thanks!")
                game_going, rew = mygame.no_thanks()

        print(q_vals)
        if player_order >= 0:
            time.sleep(0.5)
        print("-----")

    print(mygame.score())
    print(mygame.winning())

    for player in range(mygame.n_players):
        print(
            f"{mygame.get_player_tokens_int(player):<3}|{print_cards_from_one_hot(mygame.player_cards[player])}"
        )

    embedd = params[0][0]
    embedded_game_states = [0, 0, 0, 0]
    for i in range(4):
        embedded_game_states[i] = np.array([np.dot(x, embedd) for x in game_states[i]])

    for j in range(5):
        plt.title(f"Bla {j}")
        for i in range(4):
            plt.plot(time_states[i], embedded_game_states[i][:, j], label=i)
        plt.legend()
        plt.show()
