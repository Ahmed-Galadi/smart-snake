from snake_env import SnakeEnv
from dqn_agent import DQNAgent 
import time

env = SnakeEnv()
agent = DQNAgent()
agent.load("training_test.pth")
round = 0
best_score = 0
for episode in range(20):
    state = env.reset()
    while not env.is_over():
        action = agent.choose_action(state, epsilon=0)
        state, reward, over = env.step(action)
        #time.sleep(0.1)
    if best_score < env.score:
        best_score = env.score
    print(f"episode: {episode} | score: {env.score}")
print(f"BEST SCORE: {best_score}")
