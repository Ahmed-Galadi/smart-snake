import torch
import torch.nn as nn
from collections import deque
import random

class DQNAgent(nn.Module):
    def __init__(self, max_size = 10000):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(11, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4)
                )
        self.xp = deque(maxlen=max_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def store(self, state, action, reward, next_state, over):
        out = [state, action, reward, next_state, over]
        self.xp.append(out)

    def sample(self, batch_size = 64):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_overs = []
        
        # extract random batch size elements from xp (experiance)
        batchCount = 0
        if len(self.xp) >= batch_size:
            while batchCount < batch_size:
                randIndex = random.randint(0, len(self.xp) - 1)
                all_states.append(self.xp[randIndex][0])
                all_actions.append(self.xp[randIndex][1])
                all_rewards.append(self.xp[randIndex][2])
                all_next_states.append(self.xp[randIndex][3])
                all_overs.append(self.xp[randIndex][4])
                batchCount += 1
        else:
            pass
        
        # transform them to tensors for training
        statesTensor = torch.tensor(all_states, dtype=torch.float32)
        actionsTensor = torch.tensor(all_actions).unsqueeze(1)
        rewardsTensor = torch.tensor(all_rewards).unsqueeze(1)
        nextStatesTensor = torch.tensor(all_next_states, dtype=torch.float32)
        oversTensor = torch.tensor(all_overs, dtype=torch.float32).unsqueeze(1)

        return (statesTensor, actionsTensor, rewardsTensor, nextStatesTensor, oversTensor)
    
    def choose_action(self, state, epsilon):
        randf = random.random()
        if randf < epsilon:
            return random.randint(0, 3)
        else:
            stateTensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.net(stateTensor)
                action = torch.argmax(q_values).item()
                return action
    
    def train(self, batch_size = 64, gamma = 0.9):
        if len(self.xp) < batch_size:
            return
        
        # get samples
        states, actions, rewards, nextStates, overs = self.sample(batch_size)
        
        # get Q values
        q_values = self.net(states)
       
        # predict Q values curent state
        predicted = q_values.gather(1, actions.long())
        
        # compute target with the next state
        with torch.no_grad():
            nextQ = self.net(nextStates)
            maxNextQ = nextQ.max(1, keepdim=True)[0]
            target = rewards + gamma * maxNextQ * (1 - overs)

        # calculate loss and update network
        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def __len__(self):
        return len(self.xp)
