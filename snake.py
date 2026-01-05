
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
import sys

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device " + str(device))

boardSize = 10
maxTime = 400
discountRate = 0.99
GAEparam = 0.95
PPOeps = 0.1

# Network params
value_coef = 0.5
entropy_coef = 0.01
learning_rate = 1e-03
batchSize = 512

environment_name = "Snake, rectilinear"

dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

appleReward = 1
winReward = 0
loseReward = -5

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

    def validActions(self):
        horiz_act = torch.zeros(boardSize)
        vert_act = torch.zeros(boardSize)
        acts = [horiz_act, vert_act]
        for d in range(4):
            pos = shiftPos(self.head, d)
            while isValid(pos):
                if self.queryBody(pos) != -1:
                    break
                acts[d%2][pos[1-d%2]] = 1
                if pos == self.apple:
                    break
                pos = shiftPos(pos, d)
        return torch.cat([acts[0], acts[1]])

    def makeAction(self, action):
        actionOrient = action >= boardSize
        actionPosition = action % boardSize
        actionDir = actionPosition < self.head[1-actionOrient]
        actionMag = abs(actionPosition - self.head[1-actionOrient])

        d = actionOrient + actionDir*2

        init_time = self.time
        reward = -0.01

        for i in range(actionMag):
            self.editBody(self.head, d)
            self.head = shiftPos(self.head, d)
            self.time += 1

            if self.head == self.apple:
                if np.sum(self.body == -1) == 1:
                    return winReward, True
                self.randomizeApple()
                reward = appleReward * discountRate ** (self.time - init_time)
                break
            else:
                tailDir = self.queryBody(self.tail)
                self.editBody(self.tail, -1)
                self.tail = shiftPos(self.tail, tailDir)

        if np.sum(self.body == -1) == 1:
            return winReward, True
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
        arr[4, self.head[0], self.head[1]] = 5
        arr[5, self.tail[0], self.tail[1]] = 5
        arr[6, self.apple[0], self.apple[1]] = 5
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
# print(env.validActions())
# print(env.makeAction(3))
# print(env)
# print(env.validActions())
# print(env.makeAction(19))
# print(env)
# print(env.validActions())
# print(env.makeAction(5))
# print(env)
# print(env.validActions())
# print(env.makeAction(8))
# print(env)
# print(env.validActions())
# print(env.makeAction(18))
# print(env)
# print(env.validActions())
# print(env.makeAction(9))
# print(env)
# print(env.validActions())
# print(env.makeAction(19))
# print(env)
# print(env.validActions())



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(7, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 512)
        self.policy = nn.Linear(512, 20)
        self.value = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.policy(x), self.value(x)

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(7, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 5 * 5, 512)
#         self.fc2 = nn.Linear(512, 500)
#         self.fc3 = nn.Linear(512, 256)
#         self.value = nn.Linear(256, 1)

#         self.conv3 = nn.Conv2d(37, 32, 3, padding=1)
#         self.policy = nn.Conv2d(32, 1, 3, padding=1)
    
#     def forward(self, x):
#         B, _, _, _ = x.shape
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         y = self.pool(x)
#         y = torch.flatten(y, 1)
#         y = F.relu(self.fc1(y))
#         z = F.relu(self.fc3(y))
#         value = self.value(z)
#         y = F.relu(self.fc2(y))
#         comb = torch.cat([x, y.view(B, 5, 10, 10)], axis=1)
#         comb = F.relu(self.conv3(comb))
#         policy = self.policy(comb).view(B, 100)
#         return policy, value

# m = CNN()
# print(m(Environment().toTensor().reshape(1, 7, 10, 10)))

def printLineToFile(file, text):
    if file is None:
        print(text)
    else:
        with open(file, 'a') as f:
            f.write(text + '\n')

class PPO():

    def __init__(self):

        self.model = CNN().to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.gameOutput = "game.txt"
        with open(self.gameOutput, 'w') as f:
            f.write("")
        self.debugOutput = "debug.txt"
        with open(self.debugOutput, 'w') as f:
            f.write("")
        
        self.mainOutput = None
        if self.mainOutput is not None:
            with open(self.mainOutput, 'w') as f:
                f.write("")


        s = str(summary(self.model, input_size=(1, 7, 10, 10), verbose=0))
        printLineToFile(self.mainOutput, s)

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
                trajectories[i][t]['reward'], endState = new_env.makeAction(actions[index].item())
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

    # def PGUpdate(self, trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     for t in range(len(active_thread_hist)):
    #         inputs = []
    #         emp_vals = []
    #         advantages = []
    #         for i in active_thread_hist[t]:
    #             inputs.append(trajectories[i][t]['env'].toTensor())
    #             emp_vals.append(trajectories[i][t]['emp_val'])
    #             advantages.append(trajectories[i][t]['GAE'])
    #         logits, value = self.model(torch.stack(inputs).to(device))
    #         emp_vals = torch.tensor(emp_vals).to(device)
    #         advantages = torch.tensor(advantages).to(device)
    #         # advantage = (emp_vals - value.reshape((-1,))).detach()

    #         loss = (-log_prob_hist[t] * advantages 
    #                 + value_coef * (value_hist[t] - emp_vals) ** 2 
    #                 - entropy_coef * dist_hist[t].entropy()).mean()

    #         loss.backward()
    #     self.optimizer.step()
    
    def PPOupdate(self, batchSize, trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist):
        self.model.train()
        identifiers = []
        for t in range(len(active_thread_hist)):
            for index in range(len(active_thread_hist[t])):
                identifiers.append([t, index])
        random.shuffle(identifiers)
        inputs = []
        emp_vals = []
        advantages = []
        actions = []
        old_logprob = []
        valid_acts = []
        for inst_id, (t, index) in enumerate(identifiers):
            i = active_thread_hist[t][index]
            inputs.append(trajectories[i][t]['env'].toTensor())
            emp_vals.append(trajectories[i][t]['emp_val'])
            advantages.append(trajectories[i][t]['GAE'])
            a = action_hist[t][index]
            actions.append(a)
            old_logprob.append(torch.log(dist_hist[t].probs[index, a]).item())
            valid_acts.append(trajectories[i][t]['env'].validActions())
            if (inst_id % batchSize == batchSize-1 and inst_id + batchSize < len(identifiers)) or inst_id == len(identifiers)-1:

                self.optimizer.zero_grad()
                logits, value = self.model(torch.stack(inputs).to(device))

                emp_vals = torch.tensor(emp_vals).to(device)
                advantages = torch.tensor(advantages).to(device)
                actions = torch.tensor(actions).to(device)
                old_logprob = torch.tensor(old_logprob).to(device)
                valid_acts = torch.stack(valid_acts).to(device)

                logits = logits.masked_fill(valid_acts == 0, float('-inf'))

                curr_dist = Categorical(logits=logits)

                ratio = torch.exp(curr_dist.log_prob(actions) - old_logprob)
                clipped = ratio.clip(max=1+PPOeps, min=1-PPOeps)
                CLIP_loss = -torch.min(ratio * advantages, clipped * advantages)

                loss = (CLIP_loss
                         + value_coef * (value.reshape((-1,)) - emp_vals) ** 2 
                         - entropy_coef * curr_dist.entropy()).mean()
                loss.backward()
                self.optimizer.step()

                inputs = []
                emp_vals = []
                advantages = []
                actions = []
                old_logprob = []
                valid_acts = []


    def trainLoop(self, n_iter, n_threads, evalPeriod, include_accuracy=False):

        scoreHist = []
        sizeHist = []
        lengthsHist = []
        timesHist = []
        accuracyHist = []
        winHist = []
        lossHist = []
        winTimeHist = []

        printLineToFile(self.mainOutput, "Starting training, Previous iterations: " + str(self.itCount) + " Current iterations: " + str(n_iter) + " Timestamp: " + str(datetime.now()))
        printLineToFile(self.mainOutput, f'Environment: {environment_name}, boardSize: {boardSize}, maxTime: {maxTime}, loseReward: {loseReward}, discountRate: {discountRate}, GAEParam: {GAEparam}')
        printLineToFile(self.mainOutput, f'Value coef: {value_coef}, Entropy coef: {entropy_coef}, Learning rate: {learning_rate}, batchSize: {batchSize}, numThreads: {n_threads}')

        for it in range(n_iter):
            
            with open(self.debugOutput, 'a') as f:
                s = 0
                c = 0
                for param_tens in self.model.parameters():
                    s += param_tens.abs().sum()
                    c += len(param_tens.flatten())
                f.write(f"Iteration {it}, Abs net weight: {s/c}\n")

            trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist = self.generateTrajectories(n_threads)
            self.PPOupdate(batchSize, trajectories, active_thread_hist, value_hist, action_hist, log_prob_hist, dist_hist)

            # Evaluation

            scores = []
            sizes = []
            lengths = []
            times = []
            wins = []
            losses = []
            winTimes = []

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

                win = (traj[-1]['env'].body == -1).sum() == 1
                wins.append(win)
                losses.append((len(traj[-1]['env'].validDirs()) == 0) and not win)

                if win:
                    winTimes.append(elapsed_time)
            
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
            winTimeHist.append(np.array(winTimes).sum())

            self.itCount += 1
            if self.itCount % evalPeriod == 0:
                avgWinTime = "NA"
                winRate = np.array(winHist[-evalPeriod:]).mean()
                if winRate > 0:
                    avgWinTime = str(np.array(winTimeHist[-evalPeriod:]).mean() / winRate / n_threads)
                printLineToFile(self.mainOutput, "Iteration: " + str(self.itCount) + 
                            " Score: " + str(np.array(scoreHist[-evalPeriod:]).mean()) +
                            " Size: " + str(np.array(sizeHist[-evalPeriod:]).mean()) +
                            " Game Length: " + str(np.array(lengthsHist[-evalPeriod:]).mean()) + 
                            " Elapsed Time: " + str(np.array(timesHist[-evalPeriod:]).mean()) + 
                            " Wins: " + str(winRate) + 
                            " Losses: " + str(np.array(lossHist[-evalPeriod:]).mean()) + 
                            " Win Time: " + avgWinTime + 
                            (" Accuracy: " + str(np.array(accuracyHist[-evalPeriod:]).mean()) if include_accuracy else "") + 
                            " Timestamp: " + str(datetime.now()))

                with open(self.gameOutput, 'a') as f:
                    s = "----------------------------- Iteration " + str(self.itCount) + ' -----------------------------\n'
                    for index in range(n_threads):
                        s += '"################ Thread ' + str(index) + ' ################\n'
                        for t in range(len(active_thread_hist)):
                            try:
                                i = active_thread_hist[t].index(index)
                            except ValueError:
                                continue
                            s += 'Step: ' + str(t) + '\n'
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
    ppo = PPO()
else:
    assert mode == 'READ'
    with open('snake.pkl', 'rb') as file:
        ppo = pickle.load(file)

ppo.trainLoop(2000, 32, 100)
