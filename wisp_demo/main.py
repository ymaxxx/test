import pygame
from settings import *
from wisp import Wisp
from module import CodeModule

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wisp Demo")
clock = pygame.time.Clock()

# 初始化对象
wisp = Wisp((WIDTH // 2, HEIGHT // 2))
module = CodeModule((100, 100))

sprites = pygame.sprite.Group()
sprites.add(wisp)
sprites.add(module)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            wisp.target = pygame.Vector2(event.pos)
        module.handle_event(event)

    sprites.update()

    screen.fill(BG_COLOR)
    sprites.draw(screen)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()