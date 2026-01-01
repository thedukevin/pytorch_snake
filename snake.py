
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device " + str(device))

boardSize = 10
maxTime = 20
discountRate = 0.99
GAEparam = 0.95

# Network params
value_coef = 0.5
entropy_coef = 0.01
learning_rate = 1e-03



dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

appleReward = 1
winReward = 0
loseReward = 0

class Environment:
    def __init__(self):
        self.body = np.full((boardSize, boardSize), -1)
        self.head = (5, 2)
        self.tail = (5, 1)
        self.editBody(self.tail, 0)
        self.randomizeApple()
        self.time = 0
    
    def validDirs(self):
        validDirs = []
        for d in range(4):
            newHead = shiftPos(self.head, d)
            if isValid(newHead) and self.queryBody(newHead) == -1:
                validDirs.append(d)
        return validDirs

    def makeAction(self, action):
        target = (action // boardSize, action % boardSize)

        reward = -0.01
        init_time = self.time
        while target != self.head:
            dirs = self.validDirs()
            if len(dirs) == 0:
                print(self)
            assert len(dirs) > 0
            minDist = 1e+10
            action = -1
            for d in dirs:
                newHead = shiftPos(self.head, d)
                newDist = abs(newHead[0] - target[0]) + abs(newHead[1] - target[1])
                if newDist < minDist:
                    minDist = newDist
                    action = d
            assert action >= 0
            self.editBody(self.head, action)
            self.head = shiftPos(self.head, action)

            self.time += 1

            if self.head == self.apple:
                if np.sum(self.body == -1) == 1:
                    return winReward, True
                if len(self.validDirs()) == 0:
                    return loseReward, True
                self.randomizeApple()
                reward = appleReward * discountRate ** (self.time - init_time)
                break
            else:
                tailDir = self.queryBody(self.tail)
                self.editBody(self.tail, -1)
                self.tail = shiftPos(self.tail, tailDir)
            
            if len(self.validDirs()) == 0:
                return loseReward, True
            
            
        return reward, False
    
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
# print(env.makeAction(53))
# print(env)
# print(env.makeAction(81))
# print(env)
# print(env.makeAction(95))
# print(env)
# print(env.makeAction(14))
# print(env)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.policy = nn.Linear(512, 100)
        self.value = nn.Linear(512, 1)
    
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
                # print(trajectories[i][t]['env'])
                inputs.append(trajectories[i][t]['env'].toTensor())
            logits, value = self.model(torch.stack(inputs).to(device))

            # for index, i in enumerate(active_thread_hist[t]):
            #     va = trajectories[i][t]['env'].validActions().to(device)
            #     logits[index, :] = logits[index, :].masked_fill(va == 0, float('-inf'))
            
            dist = Categorical(logits=logits)

            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            new_active_threads = copy.deepcopy(active_thread_hist[t])
            for index, i in enumerate(active_thread_hist[t]):
                new_env = copy.deepcopy(trajectories[i][t]['env'])
                trajectories[i][t]['evaluation'] = value[index].item()
                trajectories[i][t]['reward'], endState = new_env.makeAction(actions[index])
                # print("Action: " + str(actions[index]))
                # print("Endstate: " + str(endState))
                if endState:
                    new_active_threads.remove(i)
                # elif t < maxTime-1:
                trajectories[i].append({
                    'env': new_env,
                    'reward': 0,
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
            if t < maxTime-1:
                active_thread_hist.append(new_active_threads)
        # print(trajectories)
        for i in range(n_threads):
            val = 0
            emp_val = 0
            for t in range(len(trajectories[i])-2, -1, -1):
                time_diff = trajectories[i][t+1]['env'].time - trajectories[i][t]['env'].time
                val = (trajectories[i][t]['reward'] 
                       + discountRate**time_diff * trajectories[i][t+1]['evaluation'] 
                       - trajectories[i][t]['evaluation']
                       + (discountRate * GAEparam) ** time_diff * val)
                trajectories[i][t]['GAE'] = val
                emp_val = trajectories[i][t]['reward'] + discountRate**time_diff * emp_val
                trajectories[i][t]['emp_val'] = emp_val
        return trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist

    def updateModel(self, trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist):
        self.model.train()
        self.optimizer.zero_grad()
        for t in range(len(active_thread_hist)):
            inputs = []
            emp_vals = []
            advantages = []
            for i in active_thread_hist[t]:
                inputs.append(trajectories[i][t]['env'].toTensor())
                emp_vals.append(trajectories[i][t]['emp_val'])
                advantages.append(trajectories[i][t]['GAE'])
            logits, value = self.model(torch.stack(inputs).to(device))
            emp_vals = torch.tensor(emp_vals).to(device)
            advantages = torch.tensor(advantages).to(device)
            # advantage = (emp_vals - value.reshape((-1,))).detach()

            loss = (-log_prob_hist[t] * advantages 
                    + value_coef * (value_hist[t] - emp_vals) ** 2 
                    - entropy_coef * dist_hist[t].entropy()).mean()

            loss.backward()
        self.optimizer.step()

    def trainLoop(self, n_iter, n_threads, evalPeriod, include_accuracy=False):

        scoreHist = []
        sizeHist = []
        lengthsHist = []
        timesHist = []
        accuracyHist = []
        winHist = []
        lossHist = []

        with open(self.mainOutput, 'a') as f:
            f.write("Starting training, Previous iterations: " + str(self.itCount) + " Current iterations: " + str(n_iter) + " Timestamp: " + str(datetime.now()) + '\n')

        for it in range(n_iter):
            trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist = self.generateTrajectories(n_threads)
            self.updateModel(trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist)

            # Evaluation

            scores = []
            sizes = []
            lengths = []
            times = []
            wins = []
            losses = []

            for traj in trajectories:
                # size = (traj[-1]['env'].body != -1).sum() + 1
                # sizes.append(size)
                score = sum(s['reward'] for s in traj[:-1])
                scores.append(score)

                size = (traj[-1]['env'].body != -1).sum() + 1
                sizes.append(size)

                game_length = len(traj)-1
                lengths.append(game_length)

                elapsed_time = traj[-1]['env'].time
                times.append(elapsed_time)

                wins.append((traj[-1]['env'].body == -1).sum() == 1)
                losses.append(len(traj[-1]['env'].validDirs()) == 0)
            
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
            sizeHist.append(np.array(sizes).mean())
            timesHist.append(np.array(times).mean())
            lengthsHist.append(np.array(lengths).mean())
            winHist.append(np.array(wins).mean())
            lossHist.append(np.array(losses).mean())

            self.itCount += 1
            if self.itCount % evalPeriod == 0:
                with open(self.mainOutput, 'a') as f:
                    f.write("Iteration: " + str(self.itCount) + 
                        " Score: " + str(np.array(scoreHist[-evalPeriod:]).mean()) +
                        " Size: " + str(np.array(sizeHist[-evalPeriod:]).mean()) +
                        " Game Length: " + str(np.array(lengthsHist[-evalPeriod:]).mean()) + 
                        " Elapsed Time: " + str(np.array(timesHist[-evalPeriod:]).mean()) + 
                        " Wins: " + str(np.array(winHist[-evalPeriod:]).mean()) + 
                        " Losses: " + str(np.array(lossHist[-evalPeriod:]).mean()) + 
                        (" Accuracy: " + str(np.array(accuracyHist[-evalPeriod:]).mean()) if include_accuracy else "") + 
                        " Timestamp: " + str(datetime.now()) + '\n')

                with open(self.gameOutput, 'a') as f:
                    s = "----------------------------- Iteration " + str(self.itCount) + ' -----------------------------\n'
                    for index in range(n_threads):
                        s += '"################ Thread ' + str(index) + ' ################\n'
                        for t in range(len(active_thread_hist)):
                            try:
                                i = active_thread_hist[t].index(index)
                            except ValueError:
                                continue
                            s += str(trajectories[index][t]['env']) + '\n'
                            s += 'Reward: ' + str(trajectories[index][t]['reward']) + '\n'
                            s += 'Emp Val: ' + str(trajectories[index][t]['emp_val']) + '\n'
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

pg.trainLoop(2000, 32, 100)