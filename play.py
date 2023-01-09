"""Load trained model weights and play no_thanks against bot"""
from sys import exit
import numpy as np

import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

import pygame
from game import NoThanks
from model import predict, init_random_params
from pygame_things.button import TextButton

# Initialize randomness
SEED = 4
key = jr.PRNGKey(SEED)
key, sbkey = jr.split(key)

# Initialize game
mygame = NoThanks(4, 11)
mygame.start_game()
input_size = len(mygame.get_things())

# Load parameters and create leaves
npz_files = np.load("params.npz")
leaves = [npz_files[npz_files.files[i]] for i in range(len(npz_files.files))]

# Get the right PyTree definition
_, temp_params = init_random_params(sbkey, (-1, input_size))
_, treedef = tree_flatten(temp_params)

# Get parameters
params = tree_unflatten(treedef, leaves)

print("player tokens", mygame.player_state[mygame.player_turn][0])
print("Center Card", mygame.center_card)
print(predict(params, mygame.get_things()).ravel())

if __name__ == "__main__":

    # Initialize pygame
    pygame.display.init()
    DISP_SIZE = (1280, 720)

    # Initialize Screen
    rect = np.array([0, 0, DISP_SIZE[0], DISP_SIZE[1]])
    screen = pygame.display.set_mode(DISP_SIZE)
    pygame.display.set_caption("No Thanks!")

    # Initialize Font
    pygame.font.init()
    helvetica_path = pygame.font.match_font("helvetica")
    font = pygame.font.Font(helvetica_path, 14)

    # Initialize Objects
    no_thanks_button = TextButton((390, 570), (200, 80), "No Thanks!")
    take_button = TextButton((690, 570), (200, 80), "Take!")
    buttons = [no_thanks_button, take_button]

    for but in buttons:
        but.screen = screen
        but.font = font

    RUNNING = True
    while RUNNING:
        screen.fill((245, 250, 245))

        pos = pygame.mouse.get_pos()
        mouse_press = pygame.mouse.get_pressed()[0] != 0

        # Button Loop
        for but in buttons:
            but.hover = but.mouse_in_box(pos)
            but.down = mouse_press and but.hover
            but.draw()

        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("quit")
                RUNNING = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("esc")
                    RUNNING = False
            if event.type == pygame.MOUSEBUTTONUP:
                for but in buttons:
                    if but.mouse_in_box(pos):
                        print(but.text)

        pygame.display.update()
        # pygame.event.clear()

    pygame.quit()
    exit()
