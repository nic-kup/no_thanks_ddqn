"""A Dueling DDQN learning to play no_thanks"""
import jax.numpy as jnp
import jax.random as jr
from jax.nn import sigmoid
import numpy.random as npr
from jax import jit, grad
from jax.tree_util import tree_map, tree_flatten
from joblib import Parallel, delayed
from game import NoThanks
from single_game import single_game
from tree_helper import lion_step, tree_zeros_like
from model import init_random_params, predict
from sample_helpers import sample_from, sample_all

if __name__ == "__main__":

    SEED = 4
    key = jr.PRNGKey(SEED)
    key, sbkey = jr.split(key)

    STEP_SIZE = 3e-4

    mygame = NoThanks(4, 11)
    mygame.start_game()
    INPUT_SIZE = len(mygame.get_things())

    print(f"Input size: {INPUT_SIZE}")

    _, params = init_random_params(sbkey, (-1, INPUT_SIZE))
    key, sbkey = jr.split(key)

    EPOCHS = 100
    MAX_INV_TEMP = 50
    experiences = []

    @jit
    def loss(params, batch, key=None):
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

        # Hardcoded 0.99 discount
        target = r + 0.97 * done * old_next_q_values_sel

        return jnp.mean(jnp.square(q_values - target))

    dloss = jit(grad(loss))

    @jit
    def convex_comb(tree1, tree2, alpha=0.9):
        """Calculates the EMA of parameters"""
        return tree_map(
            lambda x, y: alpha * x + (1 - alpha) * y,
            tree1,
            tree2,
        )

    def play_games(predict, params, num_games, inv_temp):
        """Play num_games games with given parameters and epsilon"""
        return Parallel(n_jobs=-1, backend="threading")(
            delayed(single_game)(predict, params, 0.01, inv_temp)
            for _ in range(num_games)
        )

    # |--------------------|
    # | Main training loop |
    # |--------------------|
    print("Start training")
    old_params = tree_zeros_like(params)
    momentum = tree_zeros_like(params)
    experiences = []
    inv_temp = 1
    for epoch in range(EPOCHS):
        # Decrease randomness up to point
        inv_temp = min(epoch, MAX_INV_TEMP)

        # Set old_params to params except in the beginning
        if epoch % 10 == 1 and epoch > 2:
            print("Reset old_params")
            old_params = params.copy()
            # momentum = tree_zeros_like(params)

        # Play some games with `old_params` and `params`
        list_of_new_exp = play_games(predict, params, 50, inv_temp)
        new_exp = [item for sublist in list_of_new_exp for item in sublist]

        print(len(new_exp))

        experiences = new_exp + experiences
        experiences = experiences[:30000]

        # Gradient Descent
        for _ in range(128):
            batch = sample_from(experiences, k=256)
            grad = dloss(params, batch, sbkey)
            params, momentum = lion_step(STEP_SIZE, params, grad, momentum)
            key, sbkey = jr.split(key)

        # Print progress
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            big_batch = sample_all(experiences)
            game_loss = jnp.mean(loss(params, big_batch, sbkey))
            key, sbkey = jr.split(key)

            print(
                f"{epoch:<4.0f}:  Loss: {game_loss:<9.4f}  exp_len: {len(experiences)}"
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

    while game_going:
        print("---")
        print("Player", mygame.player_turn)
        print("player tokens", mygame.player_state[mygame.player_turn][0])
        print("Center Card", mygame.center_card)
        print("Center Tokens", mygame.center_tokens)
        print("Cards left", len(mygame.cards))

        cur_player = mygame.player_turn
        state = mygame.get_things()
        q_vals = predict(params, state).ravel()

        print("q_vals", q_vals)

        if sigmoid(MAX_INV_TEMP * (q_vals[0] - q_vals[1])) > npr.random():
            game_going, rew = mygame.take_card()
            player_store[cur_player] = (state, 0, q_vals[0])
            print(f"take: {rew}")
        else:
            game_going, rew = mygame.no_thanks()
            player_store[cur_player] = (state, 1, q_vals[1])
            print(f"no_thanks: {rew}")

    print(mygame.score())
    print(mygame.winning())

    for x in mygame.get_player_state_perspective():
        print(x[0])
        print(*x[1:])

    leaves, treedef = tree_flatten(params)

    jnp.savez("params", *leaves)
