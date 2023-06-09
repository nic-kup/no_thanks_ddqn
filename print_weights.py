from jax.tree_util import tree_flatten, tree_unflatten
from model import predict, init_random_params
from custom_layers import solu
import numpy as np
import matplotlib.pyplot as plt

# Circuits I eventually intend to find:
# - The "no thanks" circuit taking one off the number of tokens
# - The "take" circuit adding the card to ones collection

if __name__ == "__main__":

    cmap = "CMRmap"

    def relu(x):
        return np.maximum(0.0, x)

    npz_files = np.load("all_params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    embedd = leaves[0]
    unbedd = leaves[-1]
    emunbedd = np.dot(embedd, unbedd)

    print(np.diag(emunbedd))

    plt.imshow(emunbedd, cmap=cmap)
    plt.show()

    for i in range(3):
        resd1 = leaves[i * 4 + 1]
        bias1 = leaves[i * 4 + 2]
        resd2 = leaves[i * 4 + 3]
        bias2 = leaves[i * 4 + 4]

        # Does this even make sense???
        rest = np.dot(np.dot(np.dot(embedd, resd1) + bias1, resd2) + bias2, unbedd)
        plt.title(f"{i+1}th MLP (don't interpret too much!)")
        plt.imshow(rest, cmap=cmap)
        plt.show()
    
    plt.title("state => value")
    plt.plot(np.dot(leaves[-5].T, unbedd).T)
    plt.plot(np.dot(leaves[-3].T, unbedd).T)
    plt.show()

    for x in leaves:
        print("---")
        print(f"Max = {np.max(x)} , Min = {np.min(x)}")
        print(x)
        if len(x.shape) == 2 and not (x.shape[0] < 5 or x.shape[1] < 5):
            plt.imshow(x, cmap=cmap)
        else:
            plt.plot(x)
        plt.show()
