"""Load trained model weights and play no_thanks against bot"""
from sys import exit
import numpy as np
import numpy.random as npr

from jax import jit
import jax.numpy as jnp

from jax.nn import sigmoid, relu, softmax
from jax.tree_util import tree_flatten, tree_unflatten
from jax.random import PRNGKey

import time

import seaborn as sns
import matplotlib.pyplot as plt

from game import NoThanks
from model import predict, init_random_params, partial_predict
from custom_layers import solu


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
    game_states = []
    time_states = [[] for i in range(4)]

    time.sleep(0.5)
    print("|-|-|-|-|")

    turn_num = 0
    while game_going:
        cur_player = mygame.player_turn
        print(f"Player {cur_player}" + ", Your Turn!" * (cur_player == player_order))
        state = mygame.get_things().reshape((1, -1))
        game_states.append(state.squeeze())
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
    embedd_b = params[0][1]

    U, S, Vh = jnp.linalg.svd(embedd)
    U_abmax = np.max(np.abs(U))
    Uembedd = U @ embedd

    sns.heatmap(np.array(game_states), cmap="GnBu")
    plt.show()

    player_game_states = [0, 0, 0, 0]
    pn_game_states = [0, 0, 0, 0]
    for i in range(4):
        player_game_states[i] = np.array(
            [np.dot(game_states[t], embedd) + embedd_b for t in time_states[i]]
        ).squeeze()
#        pn_game_states[i] = np.array(
#            [predict(params, game_states[t].reshape((1,-1)))[1].squeeze() for t in time_states[i]]
#        ).squeeze()

    for i in range(4):
        plt.title(f"{i}: Embedding")
        x = player_game_states[i]
        print(x.shape)
        pgs_maxab = np.max(np.abs(x))
        sns.heatmap(
            x, vmin=-pgs_maxab, vmax=pgs_maxab, cmap="RdYlBu"
        )
        plt.show()

        x = relu(x @ params[2][0] + params[2][1])
        plt.title(f"{i}: Softmax MLP1")
        sns.heatmap(x, cmap="GnBu")
        plt.show()

        """
        x = relu(x @ params[4][0] + params[4][1])
        plt.title(f"{i}: Softmax MLP1")
        sns.heatmap(x, cmap="GnBu")
        plt.show()

        x = relu(x @ params[6][0])
        plt.title(f"{i}: Softmax MLP1")
        sns.heatmap(x, cmap="GnBu")
        plt.show()

        xp = relu(x @ params[1][0])
        plt.title(f"{i}: Softmax MLP1")
        sns.heatmap(xp, cmap="GnBu")
        plt.show()

        sns.heatmap((xp @ params[1][1] @ embedd.T).T, cmap = "RdYlBu")
        plt.show()

        x = x + xp @ params[1][1]

        xp = relu(x @ params[2][0])
        plt.title(f"{i}: Softmax MLP2")
        sns.heatmap(xp, cmap="GnBu")
        plt.show()

        sns.heatmap((xp @ params[2][1] @ embedd.T).T, cmap = "RdYlBu")
        plt.show()

        x = x + xp @ params[2][1]
        xp = relu(x @ params[3][0])
        plt.title(f"{i}: Softmax MLP3")
        sns.heatmap(xp, cmap="GnBu")
        plt.show()

        state_diff = ((pn_game_states[i] - player_game_states[i]) @ embedd.T).squeeze()
        plt.title(f"{i}: sn+1 hat - sn")
        sns.heatmap(
            state_diff,
            cmap="RdYlBu",
            vmin = -np.max(np.abs(state_diff)),
            vmax = np.max(np.abs(state_diff)),
        )
        plt.show()
        """
