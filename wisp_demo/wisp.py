import pygame
import math

class Wisp(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.image.load("assets/wisp.png").convert_alpha()
        self.rect = self.image.get_rect(center=pos)
        self.pos = pygame.Vector2(pos)
        self.target = pygame.Vector2(pos)
        self.speed = 3

    def update(self):
        if self.pos.distance_to(self.target) > 1:
            direction = (self.target - self.pos).normalize()
            self.pos += direction * self.speed
            self.rect.center = self.pos