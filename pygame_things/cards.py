import pygame

card_white = pygame.Color(220, 210, 214)

class Cards:
    """Playing cards with numbers on them"""

    def __init__(self, location, size, number):
        self.screen = None
        self.font = None
        self.length = size[0]
        self.height = size[1]
        self.loc_x = location[0]
        self.loc_y = location[1]
        self.number = number

    def draw(self):
        pass
