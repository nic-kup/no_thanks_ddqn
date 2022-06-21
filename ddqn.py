"""A Dueling DDQN learning to play no_thanks"""
import random
import numpy as np
import numpy.random as npr

import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad
from jax.tree_util import tree_map
from jax.experimental import optimizers
from jax.experimental.stax import (
    serial,
    Sigmoid,
    Dense,
    LeakyRelu,
    parallel,
    FanOut,
)

from game import NoThanks

SEED = 4
key = jr.PRNGKey(SEED)
key, sbkey = jr.split(key)

STEP_SIZE = 0.001

mygame = NoThanks(4, 11)
mygame.start_game()
input_size = len(mygame.get_things())


@jit
def hard_max(x, axis=-1):
    return 1.0 - jnp.sign(jnp.max(x, axis=axis).reshape((-1, 1)) - x)


def Dueling():
    """Combining value and action via dueling architecture."""
    init_fun = lambda rng, input_shape: (input_shape[-1], ())

    @jit
    def apply_fun(params, inputs, **kwargs):
        discounted_action = inputs[1] - jnp.max(inputs[1], axis=-1).reshape((-1, 1))
        return inputs[0] + discounted_action

    return init_fun, apply_fun


# Try using my conv and splitting card / token from player states
# FanOut an parallel as dualistic agent
init_random_params, predict = serial(
    Dense(300),
    LeakyRelu,
    Dense(600),
    LeakyRelu,
    Dense(400),
    LeakyRelu,
    FanOut(2),
    parallel(Dense(200), Dense(200)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(50), Dense(50)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(4), Dense(8)),
    parallel(LeakyRelu, LeakyRelu),
    parallel(Dense(1), Dense(2)),
    parallel(Sigmoid, Sigmoid),
    Dueling(),
)

predict = jit(predict)

opt_init, opt_update, get_params = optimizers.adam(STEP_SIZE)
adam_step = 0
_, params = init_random_params(sbkey, (-1, input_size))
key, sbkey = jr.split(key)

opt_state = opt_init(params)

EPOCHS = 71

experiences = []
game_going = 1


def sample_from(experience_list, predict, params, old_params, k=32):
    """Batch from experiences for DDQN"""
    sample = random.sample(experience_list, k=k)

    s = jnp.array([x[0] for x in sample])
    a = jnp.array([(1 - x[1], x[1]) for x in sample])

    r = jnp.array([x[2] for x in sample])
    sp = jnp.array(np.array([x[3] for x in sample]))
    f = jnp.array([x[4] for x in sample]).reshape((-1, 1))

    x = predict(params, sp)
    arg_x = hard_max(x)

    x_target = predict(old_params, sp)

    target = r + jnp.sum(x_target * arg_x * f, axis=-1)

    return s, a, target


def sample_all(experiences, predict, params, old_params):
    """Huge sample using all experiences for DDQN"""
    s = jnp.array([x[0] for x in experiences])
    a = jnp.array([(1 - x[1], x[1]) for x in experiences])

    r = jnp.array([x[2] for x in experiences])
    sp = jnp.array([x[3] for x in experiences])
    f = jnp.array([x[4] for x in experiences]).reshape((-1, 1))

    x = predict(params, sp)
    arg_x = hard_max(x)

    x_target = predict(old_params, sp)

    target = r + jnp.sum(x_target * arg_x * f, axis=-1)

    return s, a, target


@jit
def loss(params, batch, key=None):
    """Loss function for predictions"""
    s, a, target = batch
    preds = predict(params, s)

    return jnp.mean(jnp.square(jnp.einsum("ij,ij->i", preds, a) - target))


dloss = jit(grad(loss))


@jit
def update(step, opt_state, batch, key=None):
    """Simple update using jax optimizer"""
    params = get_params(opt_state)
    return opt_update(step, dloss(params, batch, key), opt_state)


@jit
def exp_mov_average(tree1, tree2, alpha=0.9):
    """Caluclates the EMA of parameters"""
    return tree_map(
        lambda x, y: alpha * x + (1 - alpha) * y,
        tree1,
        tree2,
    )


def num_games(ep):
    if ep < 10:
        return 24
    return 8


old_params = get_params(opt_state)


print("Start training")
for epoch in range(EPOCHS):
    eps = 1.0 / (3 * (epoch + 1))
    params = get_params(opt_state)
    new_exp = []

    old_params = exp_mov_average(old_params, params, 0.8)
    if epoch % 3 == 0:
        old_params = params

    for _ in range(num_games(epoch)):
        mygame = NoThanks(4, 11)
        mygame.start_game()
        game_going = 1
        player_store = [
            (mygame.get_things_perspective(player), 1)
            for player in range(mygame.n_players)
        ]
        reward = 0.0

        while game_going:
            cur_player = mygame.player_turn
            state = mygame.get_things()
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

    new_exp = list(filter(lambda x: len(x) == 5, new_exp))
    experiences = random.sample(
        experiences, k=min(len(experiences), 20000 - len(new_exp))
    )
    experiences = experiences + new_exp

    adam_step = 0

    for _ in range(1024):
        """Gradient Descent"""
        batch = sample_from(experiences, predict, params, old_params, k=128)
        opt_state = update(adam_step, opt_state, batch, sbkey)
        key, sbkey = jr.split(key)
        adam_step += 1
        # old_params = exp_mov_average(old_params, params, 0.99)

    if epoch % 5 == 0:
        big_batch = sample_all(experiences, predict, params, old_params)
        game_loss = jnp.mean(loss(get_params(opt_state), big_batch, sbkey))
        key, sbkey = jr.split(key)

        print(
            f"{epoch:<4.0f}:  Loss: {game_loss:<9.4f} counter: {mygame.get_counter():<4} exp_len: {len(experiences)}"
        )


print("example game")

mygame = NoThanks(4, 11)
mygame.start_game()
params = get_params(opt_state)
game_going = 1
experiences = []

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
    experiences.append([*player_store[cur_player], jnp.max(q_vals)])

    if q_vals[0] > q_vals[1]:
        print("take")
        game_going, _ = mygame.take_card()
        player_store[cur_player] = (state, 0, q_vals[0])
    else:
        print("no_thanks")
        game_going, _ = mygame.no_thanks()
        player_store[cur_player] = (state, 1, q_vals[1])

print(mygame.score())
print(mygame.winning())

for x in mygame.get_player_state_perspective():
    print(x[0])
    print(*x[1:])

# leaves, treedef = tree_util.tree_flatten(params)

# jnp.savez("params", leaves, dtype=object)
