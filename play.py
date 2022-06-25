"""Load trained model weights and play no_thanks against bot"""
import numpy as np

import jax.numpy as jnp
import jax.random as jr
from game import NoThanks
from model import predict, init_random_params
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

# Initialize randomness
SEED = 4
key = jr.PRNGKey(SEED)
key, sbkey = jr.split(key)

# Initialize game
mygame = NoThanks(4, 11)
mygame.start_game()
input_size = len(mygame.get_things())

# Load parameters and create leaves
npz_files = np.load("params.npz")
leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

# Get the right PyTree definition
_, temp_params = init_random_params(sbkey, (-1, input_size))
_, treedef = tree_flatten(temp_params)

# Get parameters
params = tree_unflatten(treedef, leaves)

print("player tokens", mygame.player_state[mygame.player_turn][0])
print("Center Card", mygame.center_card)
print(predict(params, mygame.get_things()).ravel())
