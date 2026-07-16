import pygame
from pygame.locals import *
import time
import random

# screen 
WIDTH = 1000
HEIGHT = 800

# colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#  size
SIZE = 40
class Apple:
    def __init__(self, screen, snake):
        self.snake = snake
        self.image = pygame.image.load("apple.png").convert()
        self.screen = screen
        self.x = SIZE * 3 
        self.y = SIZE * 3

    def draw(self):
        self.screen.blit(self.image, (self.x, self.y))
        pygame.display.flip()
    
    def is_snake_pos(self):
        index = 0
        while index < self.snake.s_length:
            if self.x == self.snake.x[index] and self.y == self.snake.y[index]:
                return True
            index += 1
        return False

    def move(self):
        self.x = random.randint(0, 20) * SIZE
        self.y = random.randint(0, 15) * SIZE
        while self.is_snake_pos():
            self.x = random.randint(0, 20) * SIZE
            self.y = random.randint(0, 15) * SIZE


class Snake:
    def __init__(self, screen, s_length):
        #block image
        self.s_length = s_length
        self.block = pygame.image.load("pixel.png").convert()
        # block position
        self.x = [SIZE] * s_length
        self.y = [SIZE] * s_length
        self.screen = screen
        self.direction = 'right'

    def draw(self):
        self.screen.fill(GREEN)
        for i in range(self.s_length):
            self.screen.blit(self.block, (self.x[i], self.y[i]))
        pygame.display.flip()

    def move_up(self):
        if self.direction != 'down':
            self.direction = 'up'
    
    def move_down(self):
        if self.direction != 'up':
            self.direction = 'down'
 
    def move_left(self):
        if self.direction != 'right':
            self.direction = 'left'
    
    def move_right(self):
        if self.direction != 'left':
            self.direction = 'right'
    
    def get_direction(self):
        # return (up, down, left, right) one-hot
        if self.direction == 'up':
            return [1,0,0,0]
        if self.direction == 'down':
            return [0,1,0,0]
        if self.direction == 'left':
            return [0,0,1,0]
        if self.direction == 'right':
            return [0,0,0,1]
            

    def step(self):
        for i in range(self.s_length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        if self.direction == 'up':
            self.y[0] -= SIZE
        if self.direction == 'down':
            self.y[0] += SIZE
        if self.direction == 'left':
            self.x[0] -= SIZE
        if self.direction == 'right':
            self.x[0] += SIZE
        self.draw()

    def incSnakeLen(self):
        self.s_length += 1
        self.x.append(-1)
        self.y.append(-1)



