from game import NoThanks
import numpy.random as npr


def single_game(predict, params, eps, reward_factor=1.0):
    new_exp = []
    mygame = NoThanks(4, 11, reward_factor=reward_factor)
    mygame.start_game()
    game_going = 1
    player_store = [
        (mygame.get_things_perspective(player), 1) for player in range(mygame.n_players)
    ]
    reward = 0.0

    while game_going:
        cur_player = mygame.player_turn
        state = mygame.get_things()  # Get game state (from Ps perspective)
        q_vals = predict(params, state).ravel()

        new_exp.append([*player_store[cur_player], state, 1.0])

        if eps > npr.random():
            if npr.random() > 0.5:
                game_going, reward = mygame.take_card()
                player_store[cur_player] = (state, 0, reward)
            else:
                game_going, reward = mygame.no_thanks()
                player_store[cur_player] = (state, 1, reward)
        else:
            # if npr.random() * (q_vals[0] + q_vals[1]) < q_vals[0]:
            if q_vals[0] > q_vals[1]:
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
                0.0,  # final / done? 0=yes
            ]
        )
    return list(filter(lambda x: len(x) == 5, new_exp))


def single_game_end_reward(predict, params, eps):
    new_exp = []
    mygame = NoThanks(4, 11)
    mygame.start_game()
    game_going = 1
    player_store = [
        (mygame.get_things_perspective(player), 1) for player in range(mygame.n_players)
    ]
    reward = 0.0

    while game_going:
        cur_player = mygame.player_turn
        state = mygame.get_things()  # Get game state (from Ps perspective)
        q_vals = predict(params, state).ravel()

        new_exp.append([*player_store[cur_player], state, 1.0])

        if eps > npr.random():
            if npr.random() > 0.5:
                game_going, reward = mygame.take_card()
                player_store[cur_player] = (state, 0, reward)
            else:
                game_going, reward = mygame.no_thanks()
                player_store[cur_player] = (state, 1, reward)
        else:
            if q_vals[0] > q_vals[1]:
                game_going, reward = mygame.take_card()
                player_store[cur_player] = (state, 0, reward)
            else:
                game_going, reward = mygame.no_thanks()
                player_store[cur_player] = (state, 1, reward)

    winner = mygame.winning()
    # Give each player final experience so they see loss
    new_exp = list(filter(lambda x: len(x) == 5, new_exp))

    for player in range(mygame.n_players):
        new_exp.append(
            [
                player_store[player][0],  # s_t
                player_store[player][1],  # a_t
                winner[player],  # r_t
                mygame.get_things_perspective(player),  # s_{t+1}
                0.0,  # final? 0=yes
            ]
        )
    return new_exp
