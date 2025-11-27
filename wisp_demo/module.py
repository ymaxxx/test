import pygame

class CodeModule(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.image.load("assets/print_module.png").convert_alpha()
        self.rect = self.image.get_rect(topleft=pos)
        self.dragging = False
        self.offset = pygame.Vector2(0, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.offset = pygame.Vector2(self.rect.topleft) - pygame.Vector2(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.rect.topleft = pygame.Vector2(event.pos) + self.offset
