from model import predict, init_random_params
from tree_helper import lion_step
from jax.example_libraries.stax import (
    serial,
    Sigmoid,
    Dense,
    Relu,
)
from jax import jit, grad
from jax.tree_util import tree_flatten, tree_unflatten
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    npz_files = np.load("params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    key = jr.PRNGKey(4)
    INPUT_SIZE = 171

    # Rebuild model, but just to Value
    init_params, predict1 = serial(
        Dense(512),
        Relu,
        Dense(512),
        Relu,
        Dense(256),
        Relu,
        Dense(64),
        Relu,
        Dense(1),
        Sigmoid,
    )

    key, sbkey = jr.split(key)

    @jit
    def predict(params, x):
        return jnp.ravel(predict1(params, x))[0]

    # Get tree_def
    params = init_params(sbkey, (-1, INPUT_SIZE))[1]
    tree_def = tree_flatten(params)[1]
    # Set params
    params = tree_unflatten(tree_def, leaves[:10])
    # Test
    x = jnp.zeros((1,171))
    momen = jnp.zeros((1,171))

    print(predict(params, x))

    dpredict = grad(predict, 1)
    for i in range(10000):
        if i%10==0:
            print(i)
        grad = dpredict(params, x)
        x, momen = lion_step(3e-5, x, grad, momen)
        # Todo: restrict x to realistic game state
        x = jnp.maximum(0.0, x)

    print(x)

    plt.plot(x.ravel())
    plt.show()

