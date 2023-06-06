from jax.tree_util import tree_flatten, tree_unflatten
from model import predict, init_random_params
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    def relu(x):
        return np.maximum(0.0, x)

    npz_files = np.load("all_params.npz")
    leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

    embedd = leaves[0]
    unbedd = leaves[-1]
    emunbedd = np.dot(embedd, unbedd)

    print(np.diag(emunbedd))

    plt.imshow(emunbedd)
    plt.show()

    resd1 = leaves[1]
    bias1 = leaves[2]
    resd2 = leaves[3]
    bias2 = leaves[4]
    
    # Does this even make sense???
    rest = np.dot(np.dot(np.dot(embedd, resd1)+bias1, resd2)+bias2, unbedd)
    plt.title("First MLP in state space")
    plt.imshow(rest)
    plt.show()

    
    for x in leaves:
        print("---")
        print(f"Max = {np.max(x)} , Min = {np.min(x)}")
        print(x)
        if len(x.shape)==2 and not (x.shape[0]<5 or x.shape[1]<5):
            plt.imshow(x)
        else:
            plt.plot(x)
        plt.show()
