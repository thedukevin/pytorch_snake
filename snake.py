
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

np.random.seed(42)
torch.manual_seed(42)

boardSize = 10
maxTime = 50
discountRate = 0.99

dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

appleReward = 1
winReward = 5
loseReward = 0

# Network params
value_coef = 0.5
entropy_coef = 0.01

class Environment:
    def __init__(self):
        self.head = (boardSize // 2, boardSize // 2)
        self.randomizeApple()
        self.time = 0
    
    def makeAction(self, action):
        newHead = shiftPos(self.head, action)
        # while not isValid(newHead):
        #     action = np.random.randint(4)
        #     newHead = shiftPos(self.head, action)
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
        arr = np.zeros((2, boardSize, boardSize))
        arr[0, self.head[0], self.head[1]] = 1
        arr[1, self.apple[0], self.apple[1]] = 1
        return torch.tensor(arr).to(torch.float32)
    
    def bestActions(self):
        dists = []
        for d in range(4):
            newHead = shiftPos(self.head, d)
            dists.append(abs(newHead[0] - self.apple[0]) + abs(newHead[1] - self.apple[1]))
        return [i for i in range(4) if dists[i] == min(dists)]

    def __str__(self):
        s = "Time: " + str(self.time) + '\n'
        for i in range(boardSize):
            for j in range(boardSize):
                if (i, j) == self.head:
                    s += 'H'
                elif (i, j) == self.apple:
                    s += 'A'
                else:
                    s += '.'
            s += '\n'
        return s

# class Environment:
#     def __init__(self):
#         self.body = np.full((boardSize, boardSize), -1)
#         self.head = (5, 2)
#         self.tail = (5, 1)
#         self.editBody(self.tail, 0)
#         self.randomizeApple()
#         self.time = 0
    
#     def makeAction(self, action):
#         newHead = shiftPos(self.head, action)
#         if not isValid(newHead) or self.queryBody(newHead) != -1:
#             return loseReward, True
#         self.editBody(self.head, action)
#         self.head = newHead

#         reward = 0
#         if self.head == self.apple:
#             self.randomizeApple()
#             reward = appleReward
#         else:
#             tailDir = self.queryBody(self.tail)
#             self.editBody(self.tail, -1)
#             self.tail = shiftPos(self.tail, tailDir)
        
#         self.time += 1
#         return reward, self.time == maxTime or np.sum(self.body == -1) == 1
    
#     def editBody(self, pos, newElement):
#         self.body[pos[0]][pos[1]] = newElement
    
#     def queryBody(self, pos):
#         return self.body[pos[0]][pos[1]]
    
#     def randomizeApple(self):
#         openSquares = [(i, j) for i in range(boardSize) for j in range(boardSize) if self.body[i][j] == -1 and (i, j) != self.head]
#         self.apple = openSquares[np.random.randint(len(openSquares))]
    
#     def toTensor(self):
#         arr = np.zeros((11, boardSize, boardSize))
#         for i in range(4):
#             arr[i][self.body == i] = 1
#         arr[4, self.head[0], self.head[1]] = 1
#         arr[5, self.tail[0], self.tail[1]] = 1
#         arr[6, self.apple[0], self.apple[1]] = 1
#         for i in range(4):
#             newHead = shiftPos(self.head, i)
#             arr[i + 7] = not isValid(newHead) or self.queryBody(newHead) != -1
#         return torch.tensor(arr).to(torch.float32)

#     def __str__(self):
#         s = "Time: " + str(self.time) + '\n'
#         for i in range(boardSize):
#             for j in range(boardSize):
#                 if (i, j) == self.head:
#                     s += 'H'
#                 elif self.body[i][j] != -1:
#                     s += 'O'
#                 elif (i, j) == self.apple:
#                     s += 'A'
#                 else:
#                     s += '.'
#             s += '\n'
#         return s

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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 5 * 5, 40)
        self.policy = nn.Linear(40, 4)
        self.value = nn.Linear(40, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.policy(x), self.value(x)

model = CNN()
summary(model, input_size=(1, 2, 10, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

def generateTrajectories(n_threads):
    model.eval()
    trajectories = []
    active_thread_hist = [list(range(n_threads))]
    for i in range(n_threads):
        trajectories.append([{
            'env': Environment(),
            'reward': None,
            'value': None
        }])
    value_hist = []
    action_hist = []
    log_prob_hist = []
    prob_hist = []
    for t in range(maxTime):
        inputs = []
        for i in active_thread_hist[t]:
            inputs.append(trajectories[i][t]['env'].toTensor())
        logits, value = model(torch.stack(inputs))
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        new_active_threads = copy.deepcopy(active_thread_hist[t])
        for index, i in enumerate(active_thread_hist[t]):
            new_env = copy.deepcopy(trajectories[i][t]['env'])
            trajectories[i][t]['reward'], endState = new_env.makeAction(actions[index])
            if endState:
                new_active_threads.remove(i)
            else:
                trajectories[i].append({
                    'env': new_env,
                    'reward': None,
                    'value': None
                })
        value_hist.append(value.reshape((-1,)))
        action_hist.append(actions)
        log_prob_hist.append(log_prob)
        prob_hist.append(dist.probs)
        if len(new_active_threads) == 0:
            break
        active_thread_hist.append(new_active_threads)
    for i in range(n_threads):
        val = 0
        for t in range(len(trajectories[i])-1, -1, -1):
            val *= discountRate
            val += trajectories[i][t]['reward']
            trajectories[i][t]['value'] = val
    return trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, prob_hist

# print(generateTrajectories(5))

def updateModel(trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, prob_hist):
    model.train()
    optimizer.zero_grad()
    for t in range(len(active_thread_hist)):
        inputs = []
        emp_vals = []
        for i in active_thread_hist[t]:
            inputs.append(trajectories[i][t]['env'].toTensor())
            emp_vals.append(trajectories[i][t]['value'])
        logits, value = model(torch.stack(inputs))
        assert torch.allclose(value.reshape((-1,)), value_hist[t])
        dist = Categorical(logits=logits)
        emp_vals = torch.tensor(emp_vals)
        advantage = (emp_vals - value.reshape((-1,))).detach()

        loss = (-log_prob_hist[t] * advantage + value_coef * torch.pow(value_hist[t] - emp_vals, 2) - entropy_coef * dist.entropy()).mean()
        # loss = (-log_prob_hist[t] * emp_vals + value_coef * torch.pow(value.reshape((-1,)) - emp_vals, 2) - entropy_coef * dist.entropy()).mean()

        # print("Step " + str(t))
        # print(log_prob_hist[t])
        # print(advantage)
        # print(value_hist[t])
        # print(emp_vals)
        # print(dist.log_prob())
        # print(loss)

        loss.backward()
    optimizer.step()

gameOutput = "game.txt"
with open(gameOutput, 'w') as f:
    f.write("")

def trainLoop(n_iter, n_threads, evalPeriod):

    scoreHist = []
    timesHist = []
    accuracyHist = []

    print("Starting training Timestamp: " + str(datetime.now()))

    for it in range(n_iter):
        trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, prob_hist = generateTrajectories(n_threads)
        # print(trajInfo)
        updateModel(trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, prob_hist)

        # Evaluation

        scores = []
        times = []

        for traj in trajectories:
            # size = (traj[-1]['env'].body != -1).sum() + 1
            # sizes.append(size)
            score = sum(s['reward'] for s in traj)
            scores.append(score)

            survive_time = len(traj)
            times.append(survive_time)
        
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
        if it % evalPeriod == 0 and it > 0:
            print("Iteration: " + str(it) + " Score: " + str(np.array(scoreHist[-evalPeriod:]).mean()) + " Survival Time: " + str(np.array(timesHist[-evalPeriod:]).mean()) + " Accuracy: " + str(np.array(accuracyHist[-evalPeriod:]).mean()) + " Timestamp: " + str(datetime.now()))

            with open(gameOutput, 'a') as f:
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
                        s += 'Value: ' + str(trajectories[index][t]['value']) + '\n'
                        s += 'Action: ' + str(action_hist[t][i]) + '\n'
                        s += 'Prob: ' + str(prob_hist[t][i]) + '\n'
                        s += 'Net value: ' + str(value_hist[t][i]) + '\n'
                        s += '\n'
                f.write(s)
        
        # if it == 3000:
        #     import os, shutil
        #     folder = 'snake_traj'
        #     for filename in os.listdir(folder):
        #         file_path = os.path.join(folder, filename)
        #         try:
        #             if os.path.isfile(file_path) or os.path.islink(file_path):
        #                 os.unlink(file_path)
        #             elif os.path.isdir(file_path):
        #                 shutil.rmtree(file_path)
        #         except Exception as e:
        #             print('Failed to delete %s. Reason: %s' % (file_path, e))

        #     for t in range(len(active_thread_hist)):
        #         for index, i in enumerate(active_thread_hist[t]):
        #             s = trajectories[i][t]['env']
        #             plt.figure()
        #             snake = (s.body != -1) * 0.5
        #             snake[s.head[0]][s.head[1]] = 0.5
        #             snake[s.apple[0]][s.apple[1]] = 1
        #             plt.imshow(snake)
        #             plt.title(f"Reward: {trajectories[i][t]['reward']} Value: {trajectories[i][t]['value']} Action: {action_hist[t][index]}")
        #             plt.xlabel(f"Probs: {prob_hist[t][index]}")
        #             plt.savefig(f"snake_traj/state{i}_{t}")
        #             plt.close()

# trainLoop(1, 4, 10)
trainLoop(2000, 64, 100)