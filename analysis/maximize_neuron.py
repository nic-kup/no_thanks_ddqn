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

rep_label = ["center", "cur", "n", "nn", "nnn"]


if __name__ == "__main__":
    npz_files = np.load("params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    key = jr.PRNGKey(4)
    INPUT_SIZE = 171

    # Rebuild model, but just to Value

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
    x = jnp.zeros((1, 171))
    momen = jnp.zeros((1, 171))

    print(predict(params, x))

    dpredict = grad(predict, 1)
    for i in range(5000):
        if i % 1000 == 0:
            print(f"{i:<5}: {predict(params, x)}")
        grad = dpredict(params, x)
        # -grad b/c we want to maximize
        x, momen = lion_step(2e-3, x, -grad, momen, wd=0.1)
        # Todo: restrict x to realistic game state
        x = jnp.maximum(0.0, x)

    print(x)

    plt.plot(x.ravel())
    plt.show()
    for j in range(5):
        plt.plot(
            x[0][j * 34 : (j + 1) * 34 + 1],
            label=rep_label[j],
        )
        plt.legend()
    plt.show()
