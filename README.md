# No Thanks Dueling Deep Q-Network

A Dueling DDQN reinforcement learner for the game No Thanks.

A rough outline of the algorithm:
- We initiate the Neural Network.
- The network plays against itself.
- At each turn record the current state under experiences.
- Give reward to the winner at final step of game.
- Train network on all experiences using AdamW.
- - In particular, the target is the predictions of the reward of the NN with old weights. This allows us to bootstrap learning. This approach is grounded by the reward of the final turn in the game.
- Repeat.

## Organization
Run the `ddqn.py` file to train the network. Then run the file `play.py` to play against the network. The `play.py` file is not yet functional.

The `game.py` file includes all the NoThanks relevant information.
