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
        self.image = pygame.image.load("apple.png").convert()
        self.screen = screen
        self.x = SIZE * 3 
        self.y = SIZE * 3

    def draw(self):
        self.screen.blit(self.image, (self.x, self.y))
        pygame.display.flip()
         
    def move(self):
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

class Game:
    def __init__(self):
        pygame.init()
        self.running = True
        self.score = 0
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.screen.fill(GREEN)
        self.snake = Snake(self.screen, 3)
        self.apple = Apple(self.screen, self.snake)
        self.snake.draw()
        self.apple.draw()
   
    def is_collision(self, x1, y1, x2, y2):
        if x1 >= x2 and x1 < x2 + SIZE:
            if y1 >= y2 and y1 < y2 + SIZE:
                return True
        return False


    def is_wall(self):
        if self.snake.x[0] < 0 or self.snake.x[0] >= WIDTH or self.snake.y[0] < 0 or self.snake.y[0] >= HEIGHT: 
                print("is wall!")
                return True
        return False
    
    def game_core(self):
        self.snake.step()
        self.apple.draw()
        self.display_score()
        pygame.display.flip()
        
        # collision with apple
        if self.is_collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.apple.move()
            self.score += 1
            self.snake.incSnakeLen()
            print("is collission!")

        #collision with self
        for i in range(1, self.snake.s_length):
            if self.is_collision(self.snake.x[0], self.snake.y[0], self.snake.x[i], self.snake.y[i]):
                raise "Game Over!"
        # collision with wall
        if self.is_wall():
            raise "Game Over!"

    def display_score(self):
        font = pygame.font.SysFont('arial', 38)
        score = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score, (800, 0))

    def gameOver(self):
        self.screen.fill((0, 0, 0))
        font = pygame.font.SysFont('arial', 45)
        GO_text = font.render(f"Game Over!", True, RED)
        score = font.render(f"Score: {self.score}", True, BLUE)
        message = font.render(f"Press 'ENTER' to try again", True, GREEN)
        self.screen.blit(GO_text, (200, HEIGHT/4))
        self.screen.blit(score, (300, HEIGHT/3))
        self.screen.blit(message, (400, HEIGHT/2))
        pygame.display.flip()
    
    def reset(self):
        self.score = 0
        self.screen.fill(GREEN)
        self.snake = Snake(self.screen, 3)
        self.apple = Apple(self.screen, self.snake)
        self.snake.draw()
        self.apple.draw()

    def run(self):
        self.snake.draw()
        over = False
        # game loop
        while self.running:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    if event.key == K_UP:
                        self.snake.move_up()
                    if event.key == K_DOWN:
                        self.snake.move_down()
                    if event.key == K_RIGHT:
                        self.snake.move_right()
                    if event.key == K_LEFT:
                        self.snake.move_left()
                    if event.key == K_RETURN:
                        if over:
                            self.reset()
                        over = False
                elif event.type == QUIT:
                    self.running = False
            try:
                if not over:
                    self.game_core()
            except Exception as e:
                self.gameOver()
                over = True
            time.sleep(0.3)

# game
if __name__ == "__main__":
    game = Game()
    game.run()

