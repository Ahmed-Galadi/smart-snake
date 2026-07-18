from snake_env import SnakeEnv
from dqn_agent import DQNAgent

env = SnakeEnv()
agent = DQNAgent()
rounds = 0 
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995


while rounds <= 1000:
    total_rewards = 0
    state = env.reset()
    while not env.is_over():
        action = agent.choose_action(state, epsilon)
        next_state, reward, over = env.step(action)
        total_rewards += reward
        agent.store(state, action, reward, next_state, over)
        agent.train()
        state = next_state
    if rounds % 10 == 0:
        agent.sync_target()
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"{rounds}: score-> {env.score} | total rewards-> {total_rewards} | epsilon-> {epsilon}")
    rounds += 1

agent.save("training_2.pth")


