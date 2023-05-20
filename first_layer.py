from jax.tree_util import tree_flatten, tree_unflatten
from model import predict, init_random_params
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    npz_files = np.load("params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    first_layer = leaves[0]
    print(first_layer)

    for x in first_layer.T:
        print("---")
        print(f"Max = {np.max(x)} , Min = {np.min(x)}")
        print(x)
        plt.plot(x)
        plt.show()
