from snake_game import *
import random

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.score = 0
        self.over = False
        self.reward = 0
        self.state = [0] * 14
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
        self.reward -= 1
        self.snake.step()
        self.apple.draw()
        self.display_score()
        pygame.display.flip()
        
        # collision with apple
        if self.is_collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.apple.move()
            self.score += 1
            self.snake.incSnakeLen()
            self.reward += 20

        #collision with self
        if self.is_self(self.snake.x[0], self.snake.y[0]):
            self.reward -= 20
            raise Exception("Game Over!")
        # collision with wall
        if self.is_wall():
            self.reward -= 20
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
        snkHeadX = self.snake.x[0]
        snkHeadY = self.snake.y[0]
        dir = self.snake.direction
        # danger ahead
        if (snkHeadX - SIZE < 0 and dir == 'left') or (snkHeadX + SIZE >= WIDTH and dir == 'right'):
            danger[straight] = 1;
        elif (snkHeadY - SIZE < 0 and dir == 'up') or (snkHeadY + SIZE >= HEIGHT and dir == 'down'):
            danger[straight] = 1
        elif (self.is_self(snkHeadX - SIZE, snkHeadY) and dir == 'left') or (self.is_self(snkHeadX + SIZE, snkHeadY) and dir == 'right'):
            danger[straight] = 1
        elif (self.is_self(snkHeadX, snkHeadY - SIZE) and dir == 'up') or (self.is_self(snkHeadX, snkHeadY + SIZE) and dir == 'down'):
            danger[straight] = 1
        else:
            danger[straight] = 0
        # danger sides
        if dir == 'up':
            if snkHeadX - SIZE < 0 or self.is_self(snkHeadX - SIZE, snkHeadY):
                danger[left] = 1
            else:
                danger[left] = 0

            if snkHeadX + SIZE >= WIDTH or self.is_self(snkHeadX + SIZE, snkHeadY):
                danger[right] = 1
            else:
                danger[right] = 0

        elif dir == 'down':
            if snkHeadX - SIZE < 0 or self.is_self(snkHeadX - SIZE, snkHeadY):
                danger[right] = 1
            else:
                danger[right] = 0

            if snkHeadX + SIZE >= WIDTH or self.is_self(snkHeadX + SIZE, snkHeadY):
                danger[left] = 1
            else:
                danger[left] = 0

        elif dir == 'left':
            if snkHeadY - SIZE < 0 or self.is_self(snkHeadX, snkHeadY - SIZE):
                danger[right] = 1
            else:
                danger[right] = 0
            if snkHeadY + SIZE >= HEIGHT or self.is_self(snkHeadX, snkHeadY + SIZE):
                danger[left] = 1
            else:
                danger[left] = 0

        elif dir == 'right':
            if snkHeadY - SIZE < 0 or self.is_self(snkHeadX, snkHeadY - SIZE):
                danger[left] = 1
            else:
                danger[left] = 0
            if snkHeadY + SIZE >= HEIGHT or self.is_self(snkHeadX, snkHeadY + SIZE):
                danger[right] = 1
            else:
                danger[right] = 0

        return danger

    def get_free_space(self):
        head_x = self.snake.x[0]
        head_y = self.snake.y[0]
        dir = self.snake.direction
        max_cells = (WIDTH // SIZE) * (HEIGHT // SIZE)
        result = []
        r_cells = [] # [(x, y), (x, y), (x, y)] => [straigh, left, right]
        if dir == "up":
            r_cells.append((head_x, head_y - SIZE))
            r_cells.append((head_x + SIZE, head_y))
            r_cells.append((head_x - SIZE, head_y))
        if dir == "down":
            r_cells.append((head_x, head_y + SIZE))
            r_cells.append((head_x - SIZE, head_y))
            r_cells.append((head_x + SIZE, head_y))
        if dir == "left":
            r_cells.append((head_x - SIZE, head_y))
            r_cells.append((head_x, head_y - SIZE))
            r_cells.append((head_x, head_y + SIZE))
        if dir == "right":
            r_cells.append((head_x + SIZE, head_y))
            r_cells.append((head_x, head_y + SIZE))
            r_cells.append((head_x, head_y - SIZE))
        for cell in r_cells:
            if not self.is_self(cell[0], cell[1]) and not (cell[0] < 0 or cell[0] >= WIDTH or cell[1] < 0 or cell[1] >= HEIGHT):
                result.append(self.count_reachable(cell[0], cell[1]) / max_cells)
            else:
                result.append(0)
        return result
            
                

    def count_reachable(self, startX, startY):
        startGridX = startX // SIZE
        startGridY = startY // SIZE
        visited = set()
        queue = [(startGridX, startGridY)]
        count = 0
        while queue:
            (x, y) = queue.pop(0)
            # skip if visited
            if (x, y) in visited:
                continue

            # skip if out of bound (out of walls) 
            if x < 0 or x >= WIDTH//SIZE or y < 0 or y >= HEIGHT//SIZE:
                continue
            # skip if snake body
            if self.is_self(x * SIZE, y * SIZE):
                continue
            visited.add((x, y))
            count += 1
            queue.append((x+1, y))
            queue.append((x-1, y))
            queue.append((x, y+1))
            queue.append((x, y-1))
        return count

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
        free_space = self.get_free_space()
        direction = self.snake.get_direction()
        apple = self.get_apple()

        return danger + free_space + direction + apple

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
                



