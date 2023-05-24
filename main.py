"""A Dueling DDQN learning to play no_thanks"""
from random import sample
from numpy import load
import time
import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid
import numpy.random as npr
from jax import jit, grad
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from joblib import Parallel, delayed
from game import NoThanks
from single_game import single_game
from play import print_cards_from_one_hot
from tree_helper import lion_step, tree_zeros_like, convex_comb
from model import init_random_params, predict
from sample_helpers import sample_from, sample_all

if __name__ == "__main__":

    SEED = 4
    key = jr.PRNGKey(SEED)
    key, sbkey = jr.split(key)

    STEP_SIZE = 1e-5

    CONTINUE_TRAINING_RUN = True

    mygame = NoThanks(4, 11)
    mygame.start_game()
    INPUT_SIZE = len(mygame.get_things())

    print(f"Input size: {INPUT_SIZE}")

    _, params = init_random_params(sbkey, (-1, INPUT_SIZE))
    key, sbkey = jr.split(key)

    EPOCHS = 200
    RESET_EPOCH_PER = 40
    MAX_INV_TEMP = 60
    experiences = []

    @jit
    def loss(params, batch, old_params, key=None):
        """Loss function for predictions"""
        s, a, r, sn, done = batch

        # Calculate various Q-values
        new_q_values = predict(params, s)
        new_next_q_values = predict(params, sn)
        old_next_q_values = predict(old_params, sn)

        # Apply to action
        q_values = jnp.sum(new_q_values * a, axis=-1)

        next_actions = jnp.argmax(new_next_q_values, axis=-1)
        old_next_q_values_sel = jnp.take_along_axis(
            old_next_q_values, next_actions[:, None], axis=-1
        ).squeeze()

        # Hardcoded discount
        target = r + 0.98 * done * old_next_q_values_sel

        return jnp.mean(jnp.square(q_values - target))

    dloss = jit(grad(loss))

    def play_games(predict, params, old_params, num_games, inv_temp):
        """Play num_games games with given parameters and epsilon"""
        return Parallel(n_jobs=-1, backend="threading")(
            delayed(single_game)(predict, params, old_params, 0.01, inv_temp)
            for _ in range(num_games)
        )

    # |--------------------|
    # | Main training loop |
    # |--------------------|

    old_params = tree_zeros_like(params)
    momentum = tree_zeros_like(params)
    inv_temp = 1
    experiences = []
    if CONTINUE_TRAINING_RUN:
        print("Continuing last training run")

        # Load parameters and create leaves
        npz_files = load("params.npz")
        leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

        # Get the tree definition
        tree_def = tree_flatten(params)[1]

        # Set the parameters from file
        params = tree_unflatten(tree_def, leaves)
        old_params = params.copy()

        # Set inverse temp to None = infinity
        inv_temp = None
    else:
        print("Starting a new run")

    print("Start training")
    for epoch in range(EPOCHS):
        # Decrease randomness up to MAX_INV_TEMP
        if (epoch < MAX_INV_TEMP) and (not CONTINUE_TRAINING_RUN):
            inv_temp = min(epoch, MAX_INV_TEMP)
        else:
            inv_temp = None

        # Set old_params to params except in the beginning
        if epoch % RESET_EPOCH_PER == 1 and epoch > 2:
            print("old_params <- params & momentum <- 0")
            old_params = params.copy()
            momentum = tree_zeros_like(params)

        # Play some games with `old_params` and `params`
        start_time = time.time()
        list_of_new_exp = play_games(
            predict, params, old_params, 50 + 100 * (epoch == 0), inv_temp
        )
        new_exp = [item for sublist in list_of_new_exp for item in sublist]
        time_new_exp = time.time() - start_time

        experiences = new_exp + sample(
            experiences, k=min(35000 - len(new_exp), len(experiences))
        )
        # experiences = experiences[:35000]

        # Gradient Descent
        start_time = time.time()
        for _ in range(128):  # 64*256 = 16'384
            batch = sample_from(experiences, k=256)
            grad = dloss(params, batch, old_params, sbkey)
            params, momentum = lion_step(STEP_SIZE, params, grad, momentum)
            key, sbkey = jr.split(key)
        time_grad_desc = time.time() - start_time

        print(
            f"|new_exp| = {len(new_exp):<6} times: {time_new_exp:5.2f} and {time_grad_desc:5.2f}"
        )

        # Print progress
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            big_batch = sample_all(experiences)
            game_loss = jnp.mean(loss(params, big_batch, old_params, sbkey))
            key, sbkey = jr.split(key)

            print(
                f" {epoch:<4.0f}:  Loss: {game_loss:<9.4f}  exp_len: {len(experiences)}"
            )
    # EXAMPLE GAMES
    print("example game")

    mygame = NoThanks(4, 11)
    mygame.start_game()
    game_going = 1
    experiences = []
    player_store = [
        (mygame.get_things_perspective(player), 1) for player in range(mygame.n_players)
    ]

    print("|-------|")
    while game_going:
        cur_player = mygame.player_turn
        state = mygame.get_things()
        q_vals = predict(params, state.reshape((1, -1))).ravel()
        player_persp = mygame.get_current_player()[0]

        print(f"Player: {mygame.player_turn} | Tokens: {player_persp[0]}")
        print(f"Cards: {print_cards_from_one_hot(player_persp[1:])}")
        print(f"Center: Card {mygame.center_card},  Tokens {mygame.center_tokens}")
        print("Cards left", len(mygame.cards))
        print("Q_vals", q_vals)

        if q_vals[0] > q_vals[1]:
            game_going, rew = mygame.take_card()
            player_store[cur_player] = (state, 0, q_vals[0])
            print(f"take: {rew}")
        else:
            game_going, rew = mygame.no_thanks()
            player_store[cur_player] = (state, 1, q_vals[1])
            print(f"no_thanks: {rew}")

        print("-----------")

    print(mygame.score())
    print(mygame.winning())

    for x in mygame.get_player_state_perspective():
        print(f"{x[0]:<3}|{print_cards_from_one_hot(x[1:])}")

    leaves, treedef = tree_flatten(params)

    jnp.savez("params", *leaves)
