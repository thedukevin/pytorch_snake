
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
import time
from datetime import datetime
import random

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device " + str(device))

boardSize = 10
maxTime = 1200
discountRate = 0.99
GAEparam = 0.95
PPOeps = 0.1

dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

appleReward = 1
winReward = 0
loseReward = 0

# Network params
value_coef = 0.5
entropy_coef = 0.01
learning_rate = 3e-3

environment_name = 'Snake'

# class Environment:
#     def __init__(self):
#         self.head = (boardSize // 2, boardSize // 2)
#         self.randomizeApple()
#         self.time = 0
    
#     def validActions(self):
#         return torch.ones(4)
    
#     def makeAction(self, action):
#         newHead = shiftPos(self.head, action)
#         # while not isValid(newHead):
#         #     action = np.random.randint(4)
#         #     newHead = shiftPos(self.head, action)
#         if not isValid(newHead):
#             return loseReward, True
#         self.head = newHead

#         reward = 0
#         if self.head == self.apple:
#             self.randomizeApple()
#             reward = appleReward
        
#         self.time += 1
#         return reward, self.time == maxTime
    
#     def randomizeApple(self):
#         openSquares = [(i, j) for i in range(boardSize) for j in range(boardSize) if (i, j) != self.head]
#         self.apple = openSquares[np.random.randint(len(openSquares))]
    
#     def toTensor(self):
#         arr = np.zeros((2, boardSize, boardSize))
#         arr[0, self.head[0], self.head[1]] = 1
#         arr[1, self.apple[0], self.apple[1]] = 1
#         return torch.tensor(arr).to(torch.float32)
    
#     def bestActions(self):
#         dists = []
#         for d in range(4):
#             newHead = shiftPos(self.head, d)
#             dists.append(abs(newHead[0] - self.apple[0]) + abs(newHead[1] - self.apple[1]))
#         return [i for i in range(4) if dists[i] == min(dists)]

#     def __str__(self):
#         s = "Time: " + str(self.time) + '\n'
#         for i in range(boardSize):
#             for j in range(boardSize):
#                 if (i, j) == self.head:
#                     s += 'H'
#                 elif (i, j) == self.apple:
#                     s += 'A'
#                 else:
#                     s += '.'
#             s += '\n'
#         return s

class Environment:
    def __init__(self):
        self.body = np.full((boardSize, boardSize), -1)
        self.head = (5, 2)
        self.tail = (5, 1)
        self.editBody(self.tail, 0)
        self.randomizeApple()
        self.time = 0
    
    def validActions(self):
        acts = []
        for d in range(4):
            newHead = shiftPos(self.head, d)
            acts.append(int(isValid(newHead) and self.queryBody(newHead) == -1))
        return torch.tensor(acts)

    def makeAction(self, action):
        assert self.validActions()[action]
        newHead = shiftPos(self.head, action)

        if not isValid(newHead) or self.queryBody(newHead) != -1:
            return loseReward, True
        self.editBody(self.head, action)
        self.head = newHead

        reward = 0
        if self.head == self.apple:
            if np.sum(self.body) == 1:
                return winReward, True
            self.randomizeApple()
            reward = appleReward
        else:
            tailDir = self.queryBody(self.tail)
            self.editBody(self.tail, -1)
            self.tail = shiftPos(self.tail, tailDir)
        
        if (self.validActions() == 0).all():
            return loseReward, True
        
        self.time += 1
        return reward, self.time == maxTime or np.sum(self.body == -1) == 1
    
    def editBody(self, pos, newElement):
        self.body[pos[0]][pos[1]] = newElement
    
    def queryBody(self, pos):
        return self.body[pos[0]][pos[1]]
    
    def randomizeApple(self):
        openSquares = [(i, j) for i in range(boardSize) for j in range(boardSize) if self.body[i][j] == -1 and (i, j) != self.head]
        self.apple = openSquares[np.random.randint(len(openSquares))]
    
    def toTensor(self):
        arr = np.zeros((7, boardSize, boardSize))
        for i in range(4):
            arr[i][self.body == i] = 1
        arr[4, self.head[0], self.head[1]] = 1
        arr[5, self.tail[0], self.tail[1]] = 1
        arr[6, self.apple[0], self.apple[1]] = 1
        return torch.tensor(arr).to(torch.float32)

    def __str__(self):
        s = "Time: " + str(self.time) + '\n'
        for i in range(boardSize):
            for j in range(boardSize):
                if (i, j) == self.head:
                    s += 'H'
                elif self.body[i][j] != -1:
                    s += 'O'
                elif (i, j) == self.apple:
                    s += 'A'
                else:
                    s += '.'
            s += '\n'
        return s

# env = Environment()
# print(env)
# print(env.makeAction(0))
# print(env)
# print(env.makeAction(1))
# print(env.makeAction(1))
# print(env.makeAction(1))
# print(env.makeAction(1))
# print(env.makeAction(0))
# print(env.makeAction(0))
# print(env)
# print(env.makeAction(2))
# print(env.toTensor())

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2, 8, 3, padding=1)
#         self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(8 * 5 * 5, 40)
#         self.policy = nn.Linear(40, 4)
#         self.value = nn.Linear(40, 1)
#         self.dropout = nn.Dropout(0.2)
    
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         return self.policy(x), self.value(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.policy = nn.Linear(128, 4)
        self.value = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.policy(x), self.value(x)

class PolicyGradient():

    def __init__(self):

        self.model = CNN().to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.gameOutput = "game.txt"
        with open(self.gameOutput, 'w') as f:
            f.write("")
        
        self.mainOutput = "main.txt"
        s = str(summary(self.model, input_size=(1, 7, 10, 10), verbose=0))
        with open(self.mainOutput, 'w') as f:
            f.write(s + '\n')

        self.itCount = 0

    def generateTrajectories(self, n_threads):
        self.model.eval()
        trajectories = []
        active_thread_hist = [list(range(n_threads))]
        for i in range(n_threads):
            trajectories.append([{
                'env': Environment(),
                'reward': None,
                'emp_val': None,
                'evaluation': None,
                'GAE': None
            }])
        value_hist = []
        action_hist = []
        log_prob_hist = []
        dist_hist = []
        for t in range(maxTime):
            inputs = []
            for i in active_thread_hist[t]:
                # if i == 0:
                #     print(trajectories[i][t]['env'])
                inputs.append(trajectories[i][t]['env'].toTensor())
            logits, value = self.model(torch.stack(inputs).to(device))

            for index, i in enumerate(active_thread_hist[t]):
                va = trajectories[i][t]['env'].validActions().to(device)
                logits[index, :] = logits[index, :].masked_fill(va == 0, float('-inf'))
            
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            new_active_threads = copy.deepcopy(active_thread_hist[t])
            for index, i in enumerate(active_thread_hist[t]):
                new_env = copy.deepcopy(trajectories[i][t]['env'])
                trajectories[i][t]['evaluation'] = value[index].item()
                trajectories[i][t]['reward'], endState = new_env.makeAction(actions[index])
                if endState:
                    new_active_threads.remove(i)
                # else:
                trajectories[i].append({
                    'env': new_env,
                    'reward': None,
                    'emp_val': None,
                    'evaluation': 0,
                    'GAE': None
                })
            value_hist.append(value.reshape((-1,)))
            action_hist.append(actions)
            log_prob_hist.append(log_prob)
            dist_hist.append(dist)
            if len(new_active_threads) == 0:
                break
            active_thread_hist.append(new_active_threads)
        for i in range(n_threads):
            gae = 0
            emp_val = 0
            for t in range(len(trajectories[i])-2, -1, -1):
                emp_val *= discountRate
                emp_val += trajectories[i][t]['reward']
                trajectories[i][t]['emp_val'] = emp_val
                gae *= discountRate*GAEparam
                gae += (trajectories[i][t]['reward'] 
                        + discountRate * trajectories[i][t+1]['evaluation'] 
                        - trajectories[i][t]['evaluation'])
                if GAEparam == 1:
                    assert abs(gae - (emp_val - trajectories[i][t]['evaluation'])) < 1e-08
                trajectories[i][t]['gae'] = gae
        return trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist

    def updateModel(self, trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist):
        self.model.train()
        self.optimizer.zero_grad()
        for t in range(len(active_thread_hist)):
            inputs = []
            emp_vals = []
            gae = []
            for i in active_thread_hist[t]:
                inputs.append(trajectories[i][t]['env'].toTensor())
                emp_vals.append(trajectories[i][t]['emp_val'])
                gae.append(trajectories[i][t]['gae'])
            logits, value = self.model(torch.stack(inputs).to(device))

            for index, i in enumerate(active_thread_hist[t]):
                va = trajectories[i][t]['env'].validActions().to(device)
                logits[index, :] = logits[index, :].masked_fill(va == 0, float('-inf'))
            
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action_hist[t])

            emp_vals = torch.tensor(emp_vals).to(device)
            gae = torch.tensor(gae).to(device)
            # advantage = (emp_vals - value.reshape((-1,))).detach()

            assert log_prob.requires_grad
            assert not gae.requires_grad
            assert value.requires_grad
            assert not emp_vals.requires_grad
            assert dist.entropy().requires_grad
            loss = (-log_prob * gae 
                    + value_coef * torch.pow(value.reshape((-1,)) - emp_vals, 2) 
                    - entropy_coef * dist.entropy()).mean()

            loss.backward()
        self.optimizer.step()
    
    def PPOUpdate(self, batchSize, trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist):
        self.model.train()
        identifiers = []
        for t in range(len(active_thread_hist)):
            for index in range(len(active_thread_hist[t])):
                identifiers.append([t, index])
        identifiers = random.shuffle(identifiers)
        for batch_id in range(len(identifiers) // batchSize):
            self.optimizer.zero_grad()
            inputs = []
            emp_vals = []
            gae = []
            va = []
            actions = []
            old_prob = []
            for batch_el in range(batchSize):
                t, index = identifiers[batch_id*batchSize + batch_el]
                i = active_thread_hist[t][index]
                inputs.append(trajectories[i][t]['env'].toTensor())
                emp_vals.append(trajectories[i][t]['emp_val'])
                gae.append(trajectories[i][t]['gae'])
                va.append(trajectories[i][t]['env'].validActions())
                action = action_hist[t][index]
                actions.append(action_hist[t][index])
                old_prob.append(dist_hist[t][index].probs[action])
            logits, value = self.model(torch.stack(inputs).to(device))

            emp_vals = torch.tensor(emp_vals).to(device)
            gae = torch.tensor(gae).to(device)
            va = torch.tensor(va).to(device)
            actions = torch.tensor(actions).to(device)
            old_prob = torch.tensor(old_prob).to(device).detach()

            logits = logits.masked_fill(va == 0, float('-inf'))

            # for index, i in enumerate(active_thread_hist[t]):
            #     logits[index, :] = logits[index, :].masked_fill(va == 0, float('-inf'))
            
            dist = Categorical(logits=logits)
            action_probs = dist.log_prob(actions)

            # advantage = (emp_vals - value.reshape((-1,))).detach()

            assert action_probs.requires_grad
            assert not gae.requires_grad
            assert not old_prob.requires_grad
            assert value.requires_grad
            assert not emp_vals.requires_grad
            assert dist.entropy().requires_grad
            loss = (action_probs / old_prob * gae)
            clipped = loss.clip(max=1+PPOeps, min=1-PPOeps)
            
            # loss = (-log_prob * gae 
            #         + value_coef * torch.pow(value.reshape((-1,)) - emp_vals, 2) 
            #         - entropy_coef * dist.entropy()).mean()

            loss.backward()
            self.optimizer.step()

    def trainLoop(self, n_iter, n_threads, evalPeriod, include_accuracy=False):

        scoreHist = []
        timesHist = []
        accuracyHist = []
        sizeHist = []
        winHist = []
        lossHist = []

        with open(self.mainOutput, 'a') as f:
            f.write("Starting training, Previous iterations: " + str(self.itCount) + " Current iterations: " + str(n_iter) + " Timestamp: " + str(datetime.now()) + '\n')
            f.write(f'Environment: {environment_name}, boardSize: {boardSize}, maxTime: {maxTime}, discountRate: {discountRate}, GAEParam: {GAEparam}\n')
            f.write(f'Value coef: {value_coef}, Entropy coef: {entropy_coef}, Learning rate: {learning_rate}, numThreads: {n_threads}\n')

        for it in range(n_iter):
            trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist = self.generateTrajectories(n_threads)
            self.updateModel(trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist)

            # Evaluation

            scores = []
            times = []
            sizes = []
            wins = []
            losses = []

            for traj in trajectories:
                size = (traj[-1]['env'].body != -1).sum() + 1
                sizes.append(size)
                score = sum(s['reward'] for s in traj[:-1])
                scores.append(score)

                survive_time = len(traj)-1
                times.append(survive_time)

                wins.append(size == 100)
                losses.append(traj[-1]['env'].validActions().sum() == 0)
            
            if include_accuracy:
                acc = []
                for index in range(n_threads):
                    for t in range(len(active_thread_hist)):
                        try:
                            i = active_thread_hist[t].index(index)
                        except ValueError:
                            continue
                        acc.append(action_hist[t][i] in trajectories[index][t]['env'].bestActions())
                
                accuracyHist.append(np.array(acc).mean())
            
            scoreHist.append(np.array(scores).mean())
            timesHist.append(np.array(times).mean())
            sizeHist.append(np.array(sizes).mean())
            winHist.append(np.array(wins).mean())
            lossHist.append(np.array(losses).mean())

            self.itCount += 1
            if self.itCount % evalPeriod == 0:
                with open(self.mainOutput, 'a') as f:
                    f.write("Iteration: " + str(self.itCount) + 
                        " Score: " + str(np.array(scoreHist[-evalPeriod:]).mean()) +
                        " Game Length: " + str(np.array(timesHist[-evalPeriod:]).mean()) + 
                        " Size: " + str(np.array(sizeHist[-evalPeriod:]).mean()) + 
                        " Wins: " + str(np.array(winHist[-evalPeriod:]).mean()) + 
                        " Losses: " + str(np.array(lossHist[-evalPeriod:]).mean()) + 
                        (" Accuracy: " + str(np.array(accuracyHist[-evalPeriod:]).mean()) if include_accuracy else "") + 
                        " Timestamp: " + str(datetime.now()) + '\n')

                with open(self.gameOutput, 'a') as f:
                    s = "----------------------------- Iteration " + str(it) + ' -----------------------------\n'
                    for index in range(n_threads):
                        s += '"################ Thread ' + str(index) + ' ################\n'
                        for t in range(len(active_thread_hist)):
                            try:
                                i = active_thread_hist[t].index(index)
                            except ValueError:
                                continue
                            s += str(trajectories[index][t]['env']) + '\n'
                            s += 'Reward: ' + str(trajectories[index][t]['reward']) + '\n'
                            s += 'Value: ' + str(trajectories[index][t]['emp_val']) + '\n'
                            s += 'GAE: ' + str(trajectories[index][t]['gae']) + '\n'
                            s += 'Action: ' + str(action_hist[t][i]) + '\n'
                            s += 'Probs: ' + str(dist_hist[t].probs[i]) + '\n'
                            s += 'Net value: ' + str(value_hist[t][i]) + '\n'
                            s += '\n'
                    f.write(s)
                
                with open('snake.pkl', 'wb') as file:
                    pickle.dump(self, file)
            
mode = "WRITE"

if mode == 'WRITE':
    pg = PolicyGradient()
else:
    assert mode == 'READ'
    with open('snake.pkl', 'rb') as file:
        pg = pickle.load(file)

pg.trainLoop(2000, 16, 100)