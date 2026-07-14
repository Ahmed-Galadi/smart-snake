from snake_game import *
import random

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.score = 0
        self.over = False
        self.reward = 0
        self.state = [0] * 11
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
            self.reward += 10

        #collision with self
        if self.is_self(self.snake.x[0], self.snake.y[0]):
            self.reward -= 10
            raise Exception("Game Over!")
        # collision with wall
        if self.is_wall():
            self.reward -= 10
            raise Exception("Game Over!")

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
   
    def is_self(self, snakeHeadX, snakeHeadY):
        for i in range(1, self.snake.s_length):
            if self.is_collision(snakeHeadX, snakeHeadY, self.snake.x[i], self.snake.y[i]):
                return True
        return False

    def get_danger(self):
        # (danger_straight, danger_right, danger_left)
        danger = [0] * 3
        straight = 0
        right = 1
        left = 2
        # danger ahead
        if (self.snake.x[0] - SIZE < 0 and self.snake.direction == 'left') or (self.snake.x[0] + SIZE >= WIDTH and self.snake.direction == 'right'):
            danger[straight] = 1;
        elif (self.snake.y[0] - SIZE < 0 and self.snake.direction == 'up') or (self.snake.y[0] + SIZE >= HEIGHT and self.snake.direction == 'down'):
            danger[straight] = 1
        elif (self.is_self(self.snake.x[0] - SIZE, self.snake.y[0]) and self.snake.direction == 'left') or (self.is_self(self.snake.x[0] + SIZE, self.snake.y[0]) and self.snake.direction == 'right'):
            danger[straight] = 1
        elif (self.is_self(self.snake.x[0], self.snake.y[0] - SIZE) and self.snake.direction == 'up') or (self.is_self(self.snake.x[0], self.snake.y[0] + SIZE) and self.snake.direction == 'down'):
            danger[straight] = 1
        else:
            danger[straight] = 0
        # danger sides
        if self.snake.direction == 'up':
            if self.snake.x[0] - SIZE < 0 or self.is_self(self.snake.x[0] - SIZE, self.snake.y[0]):
                danger[left] = 1
            elif self.snake.x[0] + SIZE >= WIDTH or self.is_self(self.snake.x[0] + SIZE, self.snake.y[0]):
                danger[right] = 1
            else:
                danger[left] = 0
                danger[right] = 0
        elif self.snake.direction == 'down':
            if self.snake.x[0] - SIZE < 0 or self.is_self(self.snake.x[0] - SIZE, self.snake.y[0]):
                danger[right] = 1
            elif self.snake.x[0] + SIZE >= WIDTH or self.is_self(self.snake.x[0] + SIZE, self.snake.y[0]):
                danger[left] = 1
            else:
                danger[right] = 0
                danger[left] = 0

        elif self.snake.direction == 'left':
            if self.snake.y[0] - SIZE < 0 or self.is_self(self.snake.x[0], self.snake.y[0] - SIZE):
                danger[right] = 1
            elif self.snake.y[0] + SIZE >= HEIGHT or self.is_self(self.snake.x[0], self.snake.y[0] + SIZE):
                danger[left] = 1
            else:
                danger[left] = 0
                danger[right] = 0

        elif self.snake.direction == 'right':
            if self.snake.y[0] - SIZE < 0 or self.is_self(self.snake.x[0], self.snake.y[0] - SIZE):
                danger[left] = 1
            elif self.snake.y[0] + SIZE >= HEIGHT or self.is_self(self.snake.x[0], self.snake.y[0] + SIZE):
                danger[right] = 1
            else:
                danger[right] = 0
                danger[left] = 0
        return danger
    
    def get_apple(self):
        # (up, down, left, right)
        output = [0] * 4
        if self.apple.y < self.snake.y[0]:
            output[0] = 1
        if self.apple.y > self.snake.y[0]:
            output[1] = 1
        if self.apple.x < self.snake.x[0]:
            output[2] = 1
        if self.apple.x > self.snake.x[0]:
            output[3] = 1
        return output

    def get_state(self):
        danger = self.get_danger()
        direction = self.snake.get_direction()
        apple = self.get_apple()

        return danger + direction + apple

    def is_over(self):
        return self.over

    def reset(self):
        self.score = 0
        self.screen.fill(GREEN)
        self.snake = Snake(self.screen, 1)
        self.apple = Apple(self.screen, self.snake)
        self.over = False
        self.reward = 0
        self.snake.draw()
        self.apple.draw()
        return self.get_state()
    
    def step(self, action):
        self.snake.draw()
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_RETURN:
                    if self.over:
                        self.reset()
            elif event.type == QUIT:
                quit()

        if action == 0: self.snake.move_up()
        elif action == 1: self.snake.move_down()
        elif action == 2: self.snake.move_left()
        elif action == 3: self.snake.move_right()

        try:
            if not self.over:
                self.game_core()
        except Exception as e:
            #self.gameOver()
            self.over = True
        reward = self.reward
        self.reward = 0
        return self.get_state(), reward, self.over 

if __name__ == "__main__":
    env = SnakeEnv()
    state = env.reset()
    steps_count = 0
    while not env.is_over():
        steps_count += 1
        #action = agent.choose_action(state)
        action = random.randint(0, 3)
        state, reward, over = env.step(action)
        print(f"step: {steps_count} | reward: {reward}")

        time.sleep(0.1)
    print(f"Game over! Score: {env.score}, Steps: {steps_count}")
                



