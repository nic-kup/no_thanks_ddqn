"""
A script to analyze the first layer of the Neural Network with simple visualizations.
"""
from jax.tree_util import tree_flatten, tree_unflatten
from model import predict, init_random_params
import numpy as np
import matplotlib.pyplot as plt

rep_label = ["center", "cur", "n", "nn", "nnn"]

if __name__ == "__main__":
    npz_files = np.load("params_end.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]
    first_layer = leaves[0]

    # Input state has 171 entries
    # Find the layer that maxes given input
    for i in range(171):
        print(f"Input {i}")
        argmax_for_input = np.argmax(first_layer[i])
        argmin_for_input = np.argmin(first_layer[i])
        plt.plot(first_layer.T[argmax_for_input])
        plt.plot(first_layer.T[argmin_for_input])
        plt.title(f"Layer {argmax_for_input}/{argmin_for_input} max/min input {i}")
        plt.show()

        for j in range(5):
            plt.plot(
                first_layer.T[argmax_for_input][j * 34 : (j + 1) * 34 + 1],
                label=rep_label[j],
            )
            plt.legend()
        plt.show()
