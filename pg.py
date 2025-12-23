
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
from torch.distributions import Categorical
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

boardSize = 5
maxTime = 10
discountRate = 0.9

dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

appleReward = 1
winReward = 5
loseReward = -5

# Network params
value_coef = 0.5
entropy_coef = 0.1

class Environment:
    def __init__(self):
        self.head = (2, 2)
        self.randomizeApple()
        self.time = 0
    
    def makeAction(self, action):
        newHead = shiftPos(self.head, action)
        if not isValid(newHead):
            return loseReward, True
        self.head = newHead

        reward = 0
        if self.head == self.apple:
            self.randomizeApple()
            reward = appleReward
        
        self.time += 1
        return reward, self.time == maxTime
    
    def randomizeApple(self):
        openSquares = [(i, j) for i in range(boardSize) for j in range(boardSize) if (i, j) != self.head]
        self.apple = openSquares[np.random.randint(len(openSquares))]
    
    def toTensor(self):
        return torch.tensor([self.head[0], self.head[1], self.apple[0], self.apple[1]]).to(torch.float32)

    def __str__(self):
        return "Time: " + str(self.time) + ' ' + str(self.toTensor())


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.policy = nn.Linear(6, 4)
        self.value = nn.Linear(6, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.policy(x), self.value(x)

model = MLP()

# simulate rollout

env = Environment()

for t in range(maxTime):
    logits, value = model(env.toTensor())
    dist = Categorical(logits=logits)
    action = dist.sample()

    print("Env: " + str(env))
    print("Logits: " + str(logits))
    print("Dist: " + str(dist.probs))
    print("Value: " + str(value))
    print("Action: " + str(action))

    reward, endState = env.makeAction(action)

    print("Reward: " + str(reward))
    print()

    if endState:
        break