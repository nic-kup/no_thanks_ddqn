from jax.tree_util import tree_flatten, tree_unflatten
from model import predict, init_random_params
import numpy as np
import matplotlib.pyplot as plt

npz_files = np.load("params.npz")
leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

for x in leaves:
    print("---")
    print(np.max(x))
    print(x)
    plt.plot(x)
    plt.show()
