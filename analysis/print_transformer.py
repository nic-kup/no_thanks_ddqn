from jax.tree_util import tree_flatten, tree_unflatten
from jax import jit
import jax.numpy as jnp
from model import predict, init_random_params
from custom_layers import solu
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Circuits I eventually intend to find:
# - The "no thanks" circuit taking one off the number of tokens
# - The "take" circuit adding the card to ones collection

if __name__ == "__main__":
    cmap = "RdYlBu"
    
    @jit
    def relu(x):
        return jnp.maximum(0.0, x)

    npz_files = np.load("all_params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    embedd = relu(leaves[0]) @ leaves[1]

    sns.heatmap(embedd @ leaves[2], cmap=cmap)
    plt.show()
    sns.heatmap(embedd @ leaves[3], cmap=cmap)
    plt.show()
    sns.heatmap(embedd @ leaves[4], cmap=cmap)
    plt.show()

    for i, x in enumerate(leaves):
        print(f"{i}: ----------")
        print(f"Max = {np.max(x)} , Min = {np.min(x)}")
        x_absmax = np.max(np.abs(x))
        if len(x.shape) == 2 and not (x.shape[0] < 5 or x.shape[1] < 5):
            sns.heatmap(x, cmap=cmap, vmin=-x_absmax, vmax=x_absmax)
        else:
            plt.plot(x)
        plt.show()
