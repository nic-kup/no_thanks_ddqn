"""Contains code for automatic game playing"""
from game import NoThanks
import numpy.random as npr
from jax.nn import sigmoid


def single_game(predict, param_list, reward_factor=1.0, inv_temp=5.0):
    """Given `predict` function automatically plays no thanks"""
    new_exp = []
    mygame = NoThanks(4, 11, reward_factor=reward_factor)
    mygame.start_game()
    game_going = 1
    player_param = npr.randint(0, len(param_list), size=4)
    player_store = [
        (mygame.get_things_perspective(player), 1) for player in range(mygame.n_players)
    ]
    reward = 0.0

    while game_going:
        cur_player = mygame.player_turn

        # Get game state (from Ps perspective)
        state = mygame.get_things()
        q_vals = predict(
            param_list[player_param[cur_player]], state.reshape((1, -1))
        ).ravel()

        new_exp.append([*player_store[cur_player], state, 1.0])

        take_q_val = 0.0
        if inv_temp == None:
            take_q_val = q_vals[0] > q_vals[1]
        else:
            take_q_val = sigmoid(inv_temp * (q_vals[0] - q_vals[1]))

        if take_q_val > npr.random():
            game_going, reward = mygame.take_card()
            player_store[cur_player] = (state, 0, reward)
        else:
            game_going, reward = mygame.no_thanks()
            player_store[cur_player] = (state, 1, reward)

    winner = mygame.winning()
    # Give each player final experience so they see loss
    for player in range(mygame.n_players):
        new_exp.append(
            [
                player_store[player][0],  # s_t
                player_store[player][1],  # a_t
                winner[player],  # r_t
                mygame.get_things_perspective(player),  # s_{t+1}
                0.0,  # game_going? No
            ]
        )
    return list(filter(lambda x: len(x) == 5, new_exp))
