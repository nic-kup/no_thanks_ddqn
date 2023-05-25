from jax.tree_util import tree_flatten, tree_unflatten
from model import predict, init_random_params
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    def relu(x):
        return np.maximum(0.0, x)

    npz_files = np.load("params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    weights_to_V = [4, 6, 8]
    # Beware of interpretation
    thing = np.dot(leaves[0], leaves[2])
    for k in weights_to_V:
        thing = np.dot(thing, leaves[k])
    plt.title("Dot of all dense matricies")
    plt.plot(thing)
    plt.show()

    for x in leaves:
        print("---")
        print(f"Max = {np.max(x)} , Min = {np.min(x)}")
        print(x)
        plt.plot(x)
        plt.show()
