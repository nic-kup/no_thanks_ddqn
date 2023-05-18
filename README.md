# No Thanks Dueling Deep Q-Network

A Dueling DDQN reinforcement learner for the game No Thanks.

A rough outline of the algorithm:
- We initiate the Neural Network.
- The network plays against itself.
- At each turn record the current state under experiences.
- Give reward to the winner at final step of game.
- Train network on all experiences using -AdamW- Lion.
- - In particular, the target is the predictions of the reward of the NN with old weights. This allows us to bootstrap learning. This approach is grounded by the reward of the final turn in the game.
- Repeat.

## Future Work
The goal of this repo is two-fold, first to familiarize myself more with deep reinforcement learning through training an agent on a simple game with sparse rewards. Second to gain expertise in analyzing neural network weights. For now this is restricted to the `print_weights.py` file, however more is to come.


## Organization
Run the `main.py` file to train the network. Then run the file `play.py` to play against the network. 

The `game.py` file includes all the NoThanks relevant information.

I coded custom layers in the `custom_layer.py` file. 
