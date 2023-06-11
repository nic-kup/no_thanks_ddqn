from jax.tree_util import tree_flatten, tree_unflatten
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

    def relu(x):
        return np.maximum(0.0, x)

    npz_files = np.load("all_params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    embedd = leaves[0]
    unbedd = leaves[-1]

    U, S, Vh = jnp.linalg.svd(embedd)

    print(U.shape)
    U_abmax = np.max(np.abs(U))
    sns.lineplot(S)
    plt.show()

    sns.heatmap(U, cmap=cmap, vmin=-U_abmax, vmax=U_abmax)
    plt.show()

    emunbedd = np.dot(embedd, unbedd)
    enun_max = np.max(np.abs(emunbedd))

    print(np.diag(emunbedd))

    sns.heatmap(emunbedd, vmin=-enun_max, vmax=enun_max, cmap=cmap)
    plt.show()

    for i in range(1):
        resd1 = leaves[i * 2 + 1]
        resd2 = leaves[i * 2 + 2]

        # Does this even make sense???
        rest = np.dot(np.dot(np.dot(embedd, resd1), resd2), unbedd)
        rest_avmax = np.max(np.abs(rest))
        plt.title(f"{i+1}th MLP (don't interpret too much!)")
        plt.imshow(rest, cmap=cmap, vmin=-rest_avmax, vmax=rest_avmax)
        plt.show()

    #    plt.title("state => value")
    #    plt.plot(np.dot(leaves[-5].T, unbedd).T)
    #    plt.plot(np.dot(leaves[-3].T, unbedd).T)
    #    plt.show()

    for i, x in enumerate(leaves):
        print(f"{i}: ----------")
        print(f"Max = {np.max(x)} , Min = {np.min(x)}")
        x_absmax = np.max(np.abs(x))
        if len(x.shape) == 2 and not (x.shape[0] < 5 or x.shape[1] < 5):
            plt.imshow(x, cmap=cmap, vmin=-x_absmax, vmax=x_absmax)
        else:
            plt.plot(x)
        plt.show()
