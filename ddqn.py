"""A Dueling DDQN learning to play no_thanks"""

import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad
from jax.tree_util import tree_map, tree_flatten
from joblib import Parallel, delayed

from game import NoThanks
from play_one_game import single_game
from tree_helper import lion_step, tree_zeros_like
from model import init_random_params, predict
from sample_helpers import sample_from, sample_all

SEED = 4
key = jr.PRNGKey(SEED)
key, sbkey = jr.split(key)

STEP_SIZE = 3e-4

mygame = NoThanks(4, 11)
mygame.start_game()
INPUT_SIZE = len(mygame.get_things())


adam_step = 0
_, params = init_random_params(sbkey, (-1, INPUT_SIZE))
key, sbkey = jr.split(key)


EPOCHS = 41

experiences = []
game_going = 1


@jit
def loss(params, batch, key=None):
    """Loss function for predictions"""
    s, a, target = batch
    preds = predict(params, s)

    return jnp.mean(-target * jnp.einsum("ij,ij->i", jnp.log(preds), a))
    # Old MSE loss
    # return jnp.mean(jnp.square(jnp.einsum("ij,ij->i", preds, a) - target))


dloss = jit(grad(loss))


@jit
def convex_comb(tree1, tree2, alpha=0.9):
    """Caluclates the EMA of parameters"""
    return tree_map(
        lambda x, y: alpha * x + (1 - alpha) * y,
        tree1,
        tree2,
    )


@jit
def sq_distance(tree1, tree2):
    """A distance between parameters for tracking change"""
    return sum(
        tree_flatten(
            tree_map(
                lambda x, y: jnp.square(x - y),
                tree1,
                tree2,
            )
        )[0]
    )


def num_games(ep):
    """Number of games to play in given epoch"""
    return 35


adam_step = 0
momentum = tree_zeros_like(params)
old_params = convex_comb(params.copy(), tree_zeros_like(params))

print("Start training")
for epoch in range(EPOCHS):
    eps = 2.0 / (4.0 * (epoch + 2.0))
    new_exp = []

    # Move old parameters closer to new parameters
    old_params = convex_comb(old_params, params, 0.9)

    # Play some games with `old_params`
    list_of_new_exp_old = Parallel(n_jobs=-1, backend="threading")(
        delayed(single_game)(predict, old_params, eps) for _ in range(num_games(epoch))
    )

    # Play some games with `params`
    list_of_new_exp = Parallel(n_jobs=-1, backend="threading")(
        delayed(single_game)(predict, old_params, eps) for _ in range(num_games(epoch))
    )
    list_of_new_exp = list_of_new_exp + list_of_new_exp_old

    # List mgmt
    new_exp = [item for sublist in list_of_new_exp for item in sublist]
    if epoch == 0:
        print(new_exp[:2])
    print(len(new_exp))

    # Randomly delete old exps with bias relative to reward
    #    max_experience_length = 5000
    #    if len(experiences) + len(new_exp) >= max_experience_length:
    #        priorities = np.array([0.5 + exper[2] for exper in experiences])
    #        priorities /= sum(priorities)
    #        indices = npr.choice(len(experiences), max_experience_length - len(new_exp), p=priorities)
    #        experiences = [experiences[idx] for idx in indices]
    experiences = new_exp  # + experiences

    for _ in range(64):
        """Gradient Descent"""
        batch = sample_from(experiences, predict, params, old_params, k=64)
        grad = dloss(params, batch, sbkey)
        params, momentum = lion_step(STEP_SIZE, params, grad, momentum)
        key, sbkey = jr.split(key)
        adam_step += 1

    if epoch % 5 == 0:
        big_batch = sample_all(experiences, predict, params, old_params)
        game_loss = jnp.mean(loss(params, big_batch, sbkey))
        key, sbkey = jr.split(key)

        print(f"{epoch:<4.0f}:  Loss: {game_loss:<9.4f}  exp_len: {len(experiences)}")


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

leaves, treedef = tree_flatten(params)

jnp.savez("params", *leaves)
