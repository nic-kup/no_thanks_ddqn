""" Create button object."""
import pygame

down_red = pygame.Color(200, 20, 25)
hover_red = pygame.Color(170, 21, 25)
mid_red = pygame.Color(140, 25, 25)
grey = pygame.Color(40, 30, 30)
black = pygame.Color(10, 10, 10)


class TextButton:
    """Generate a Button with text"""

    def __init__(self, location, size, text):
        """Initialize our slider"""
        self.screen = None
        self.font = None
        self.length = size[0]
        self.height = size[1]
        self.loc_x = location[0]
        self.loc_y = location[1]
        self.text = text
        self.hover = False
        self.down = False

    def draw(self):
        """Draw the button"""
        if self.down:
            pygame.draw.rect(
                self.screen, down_red, pygame.Rect(self.loc_x, self.loc_y, self.length, self.height)
            )
        elif self.hover:
            pygame.draw.rect(
                self.screen, hover_red, pygame.Rect(self.loc_x, self.loc_y, self.length, self.height)
            )
        else:
            pygame.draw.rect(
                self.screen, mid_red, pygame.Rect(self.loc_x, self.loc_y, self.length, self.height)
            )
        text_size = self.font.size(self.text)
        my_text = self.font.render(self.text, True, black)
        self.screen.blit(my_text, (
            self.loc_x + 0.5 * (self.length - text_size[0]),
            self.loc_y + 0.5 * (self.height - text_size[1]),
            ))
        self.down = False
        self.hover = False

    def mouse_in_box(self, mouse_pos):
        """Is mouse_pos in box"""
        if 0.0 < mouse_pos[0] - self.loc_x < self.length:
            if 0.0 < mouse_pos[1] - self.loc_y < self.height:
                return True
        return False
