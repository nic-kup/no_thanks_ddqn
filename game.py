"""Contains a class to describe the game no thanks"""
import numpy as np
import numpy.random as npr


class NoThanks:
    """Game of NoThanks"""

    def __init__(
        self,
        n_players,
        init_tokens_per_player=11,
        low_card=3,
        high_card=35,
        removed_cards=9,
        reward_factor=1.0,
    ):
        """Initialization"""
        self.n_players = n_players
        self.cards = np.array(list(range(low_card, high_card + 1)))

        self.high_card = high_card
        self.low_card = low_card
        self.n_cards = len(self.cards)

        self.removed_cards = removed_cards
        self.init_tokens_per_player = 11
        self.reward_factor = reward_factor

        self.player_state = np.zeros((n_players, self.n_cards + 1), dtype=int)
        self.player_state[:, 0] = self.init_tokens_per_player
        self.player_state = self.player_state.tolist()

        self.center_card = -1
        self.center_tokens = 0
        self.true_turn = 0
        self.turn = 0

        self.no_take_counter = 0

    @property
    def player_turn(self):
        return self.turn % self.n_players

    def start_game(self):
        """Starts the game. Shuffles cards and deals first"""
        npr.shuffle(self.cards)
        self.cards = list(self.cards)
        self.cards = self.cards[self.removed_cards :]

        self.center_card = self.cards.pop()

    def get_player_state_perspective(self, k=0):
        return self.player_state[k : self.n_players] + self.player_state[0:k]

    def get_current_player(self):
        return self.get_player_state_perspective(self.turn % self.n_players)

    def get_current_player_flat(self):
        return [k for y in self.get_current_player() for k in y]

    def get_player_state_flat(self, k=0):
        return [i for y in self.get_player_state_perspective(k) for i in y]

    @property
    def one_hot_center_card(self):
        one_hot = np.zeros(self.high_card - self.low_card + 1)
        one_hot[self.center_card - self.low_card] = 1.0
        return one_hot

    def take_card(self, punish=False):
        """current player takes center card"""
        self.no_take_counter -= 1
        reward = 0.0

        old_score = self.score_single(self.player_state[self.player_turn])

        self.player_state[self.player_turn][self.center_card - self.low_card + 1] = 1
        self.player_state[self.player_turn][0] += self.center_tokens
        self.center_tokens = 0

        new_score = self.score_single(self.player_state[self.player_turn])

        # Lower score is better, higher reward is better
        reward = old_score - new_score

        reward *= self.reward_factor
        self.true_turn += 1

        if len(self.cards) == 0:
            # Check if we end the game
            return (0, reward - punish)

        self.center_card = self.cards.pop()
        return (1, reward - punish)

    def no_thanks(self):
        """Current player no thanks"""
        self.no_take_counter += 1

        # If you have no tokens you have to take the card
        if self.player_state[self.player_turn][0] == 0:
            return self.take_card(True)
        self.player_state[self.player_turn][0] += -1
        self.center_tokens += 1
        self.turn += 1
        self.true_turn += 1
        return (1, -1.0 * self.reward_factor)

    def score(self):
        """Return cur scores of players"""
        scores = []
        for player in self.player_state:
            scores.append(self.score_single(player))
        return scores

    def winning(self):
        """Return winner"""
        scores = [0.0] * self.n_players
        inds = np.argsort(self.score())
        scores[inds[0]] = 3.0
        scores[inds[1]] = 1.0
        scores[inds[2]] = -1.0
        scores[inds[3]] = -2.0
        return scores

    def score_single(self, player):
        """Return score for single player"""
        score = -player[0]
        number = self.low_card
        prev = False
        for x in player[1:]:
            if not prev:
                score += x * number
            prev = x
            number += 1

        return score

    def get_things(self):
        """Get game state from cur_player perspective"""
        return np.concatenate(
            (
                np.array(self.center_tokens).reshape(-1),
                self.one_hot_center_card,
                self.get_current_player_flat(),
            )
        )

    def get_things_perspective(self, k=0):
        """Get game state from k's perspective"""
        return np.concatenate(
            (
                np.array(self.center_tokens).reshape(-1),
                self.one_hot_center_card,
                self.get_player_state_flat(k),
            )
        )

    def get_counter(self):
        return self.no_take_counter


if __name__ == "__main__":
    myGame = NoThanks(4, 11)
    myGame.start_game()

    print("score", myGame.score())

    print("Center card", myGame.center_card)
    print("Cards left", len(myGame.cards))

    print("no thanks")
    myGame.no_thanks()

    print("take")
    myGame.take_card()

    print("Center card", myGame.center_card)
    print("Cards left", len(myGame.cards))

    print("Score", myGame.score())

    print("--")
    print(len(myGame.get_things()))
    print(myGame.get_things())
