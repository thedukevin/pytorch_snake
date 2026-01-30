
"""

Running on cloud:
nohup python snake.py WRITE 10000 32 100 100 100 &


"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
from torch.distributions import Categorical
import copy
import pickle
import numpy as np
import time
from datetime import datetime
import random
import sys
from termcolor import colored

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device " + str(device))

boardSize = 30
maxEpisodeTime = 100
maxRolloutTime = 3000
discountRate = 0.995
GAEparam = 0.98
PPOeps = 0.2

checkpointSetback = 0
replacementProb = None
numReplacements = 50
transitionDecay = 0.95
advanceProb = 0

# Network params
# Note that having more actions doesn't necessarily mean you use a lower entropy coef.
value_coef = 0.5
entropy_coef = 0.03
learning_rate = 1e-03
batchSize = 256

anneal_winRate = 1
anneal_entropy_coef = 0.005
anneal_learning_rate = 3e-04

environment_name = "Snake rectilinear remove-forced-moves"

dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

appleReward = 0.2
winReward = 1
loseReward = -1
actionReward = 0

class Environment:
    def __init__(self):
        self.body = np.full((boardSize, boardSize), -1)
        self.pathTime = np.zeros((boardSize, boardSize))
        self.head = (boardSize//2, 2)
        self.tail = (boardSize//2, 1)
        self.body[self.tail] = 0
        self.pathTime[self.head] = 0
        self.pathTime[self.tail] = -1
        self.randomizeApple()
        self.time = 0

        self.potential = self.getPotential()
    
    def validDirs(self):
        validDirs = []
        for d in range(4):
            newHead = shiftPos(self.head, d)
            if isValid(newHead) and self.body[newHead] == -1:
                validDirs.append(d)
        return validDirs

    def validActions(self):
        horiz_act = torch.zeros(boardSize)
        vert_act = torch.zeros(boardSize)
        acts = [horiz_act, vert_act]
        for d in range(4):
            pos = shiftPos(self.head, d)
            while isValid(pos):
                if self.body[pos] != -1:
                    break
                acts[d%2][pos[1-d%2]] = 1
                if pos == self.apple:
                    break
                pos = shiftPos(pos, d)
        return torch.cat([acts[0], acts[1]]).to(torch.bool)

    def primitiveAction(self, d): # returns whether apple was acheived
        assert d in self.validDirs()
        self.body[self.head] = d
        self.head = shiftPos(self.head, d)
        self.time += 1
        self.pathTime[self.head] = self.time

        if self.head == self.apple:
            self.randomizeApple()
            return 1
        else:
            tailDir = self.body[self.tail]
            self.body[self.tail] = -1
            self.tail = shiftPos(self.tail, tailDir)
            return 0
    
    def getPotential(self): # This takes up a lot of run time for some reason
        return 0
        # self.potential = 0
        # visited = np.full((boardSize, boardSize), False)

        # tailVisible = False
        # appleDist = None

        # queue = np.array([self.head])
        # visited[self.head] = True
        # for step in range(boardSize**2):
        #     next_queue = []
        #     for d in range(4):
        #         next_queue.append(queue + np.array([dir[d]]))
            
        #     next_queue = np.concatenate(next_queue)
        #     next_queue = next_queue[((0 <= next_queue) & (next_queue < boardSize)).all(axis=1)]

        #     x = next_queue[:, 0] * boardSize + next_queue[:, 1]
        #     x = np.unique(x)
        #     next_queue = np.stack([x // boardSize, x % boardSize], axis=1)

        #     if (next_queue == np.array([self.tail])).all(axis=1).any() and step > 0:
        #         tailVisible = True
        #     next_queue = next_queue[(self.body[next_queue[:, 0], next_queue[:, 1]] == -1) & ~visited[next_queue[:, 0], next_queue[:, 1]]]

        #     if len(next_queue) == 0:
        #         break
        #     visited[next_queue[:, 0], next_queue[:, 1]] = True
        #     if (next_queue == np.array([self.apple])).all(axis=1).any() and appleDist is None:
        #         appleDist = step
        #     if tailVisible and appleDist:
        #         break
        #     queue = next_queue

        # # queue = [self.head]
        # # visited[self.head] = True
        # # finished = False
        # # for step in range(boardSize**2):
        # #     next_queue = []
        # #     for pos in queue:
        # #         for d in range(4):
        # #             newPos = shiftPos(pos, d)
        # #             if isValid(newPos) and self.queryBody(newPos) == -1 and not visited[newPos]:
        # #                 next_queue.append(newPos)
        # #                 visited[newPos] = True
        # #             if newPos == self.tail and step > 0:
        # #                 tailVisible = True
        # #             if newPos == self.apple and appleDist is None:
        # #                 appleDist = step
        # #             if tailVisible and appleDist:
        # #                 finished = True
        # #     queue = next_queue
        # #     if len(queue) == 0 or finished:
        # #         break
        
        # # a, b = self.getPotentialAlt()
        # # assert a == tailVisible and b == appleDist
        
        # tailVisiblePot = (1 - tailVisible) * -0.5
        # appleDistPot = 0 if appleDist is None else 0.05 * 0.98 ** appleDist

        # return tailVisiblePot + appleDistPot

    def makeAction(self, action):
        actionOrient = action >= boardSize
        actionPosition = action % boardSize
        actionDir = actionPosition < self.head[1-actionOrient]
        actionMag = abs(actionPosition - self.head[1-actionOrient])

        d = actionOrient + actionDir*2

        init_time = self.time
        init_pot = self.potential
        reward = actionReward

        for i in range(actionMag):
            isApple = self.primitiveAction(d)
            reward += isApple * appleReward * discountRate ** (self.time - init_time)
                
        while len(self.validDirs()) == 1:
            isApple = self.primitiveAction(self.validDirs()[0])
            reward += isApple * appleReward * discountRate ** (self.time - init_time)

        if np.sum(self.body == -1) == 1:
            return winReward, True
        if len(self.validDirs()) == 0:
            return loseReward, True
        
        self.potential = self.getPotential()

        return reward + self.potential * discountRate ** (self.time - init_time) - init_pot, False
    
    # def editBody(self, pos, newElement):
    #     self.body[pos[0]][pos[1]] = newElement
    
    # def queryBody(self, pos):
    #     return self.body[pos[0]][pos[1]]
    
    def randomizeApple(self):
        openSquares = [(i, j) for i in range(boardSize) for j in range(boardSize) if self.body[i][j] == -1 and (i, j) != self.head]
        if len(openSquares) == 0:
            return
        self.apple = openSquares[np.random.randint(len(openSquares))]
    
    def toTensor(self):
        arr = np.zeros((8, boardSize, boardSize))
        for i in range(4):
            arr[i][self.body == i] = 1
        arr[4, self.head[0], self.head[1]] = boardSize / 2
        arr[5, self.tail[0], self.tail[1]] = boardSize / 2
        arr[6, self.apple[0], self.apple[1]] = boardSize / 2
        arr[7] = np.where(self.body != -1, 0.98 **(self.pathTime-self.pathTime[self.tail]), 0)
        return torch.tensor(arr).to(torch.float32)

    def __str__(self):
        s = "Time: " + str(self.time) + '\n'
        s += "Head: " + str(self.head) + '\n'
        s += "Apple: " + str(self.apple) + '\n'
        s += "Size: " + str((self.body != -1).sum() + 1) + '\n'
        s += "Potential: " + str(self.potential) + '\n'
        grid = []
        for i in range(2*boardSize-1):
            grid.append([' '] * (2*boardSize-1))
        for i in range(boardSize):
            for j in range(boardSize):
                if (i, j) == self.head:
                    grid[2*i][2*j] = 'H'
                elif self.body[i][j] != -1:
                    d = self.body[i][j]
                    grid[2*i][2*j] = 'O'
                    grid[2*i + dir[d][0]][2*j + dir[d][1]] = '-' if d%2 == 0 else '|'
                elif (i, j) == self.apple:
                    grid[2*i][2*j] = 'A'
                else:
                    grid[2*i][2*j] = '.'
        s += '\n'.join(''.join(line) for line in grid)
        return s

# env = Environment()
# print(env)
# print(env.makeAction(12))
# print(env)
# print(env.makeAction(33))
# print(env)
# print(env.makeAction(18))
# print(env.makeAction(44))
# print(env)
# print(env.toTensor()[7])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(7, 16, 3, padding=1)
        # self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        # self.conv4 = nn.Conv2d(16, 32, 4, padding=1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        # self.policy = nn.Linear(1024, 60)
        # self.value = nn.Linear(1024, 1)

        self.conv1 = nn.Conv2d(8, 16, 5, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.policy = nn.Linear(1024, 60)
        self.value = nn.Linear(1024, 1)

        # self.conv1 = nn.Conv2d(7, 32, 3, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32 * 5 * 5, 512)
        # self.policy = nn.Linear(512, 20)
        # self.value = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.policy(x), self.value(x)

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
        self.entropy_coef = entropy_coef
        self.annealed = False

        self.gameOutput = "game.txt"
        with open(self.gameOutput, 'w') as f:
            f.write("")
        self.debugOutput = "debug.txt"
        with open(self.debugOutput, 'w') as f:
            f.write("")
        
        self.mainOutput = "main.txt"
        if self.mainOutput is not None:
            with open(self.mainOutput, 'w') as f:
                f.write("")


        s = str(summary(self.model, input_size=(1, ) + Environment().toTensor().shape, verbose=0))
        printLineToFile(self.mainOutput, s)

        self.itCount = 0

        self.checkpoints = [None] * int(boardSize**2)
        self.furthestCheckpoint = 2
        self.transitionPerProgress = [0] * int(boardSize**2)
        self.transitionLengths = [0] * int(boardSize**2)
        self.transitionPerProgress[2] = 1
        self.transitionLengths[2] = 1

    def generateTrajectories(self, n_threads):
        self.model.eval()
        trajectories = []
        active_thread_hist = [list(range(n_threads))]

        # activeCheckpoints = [i for i in range(boardSize**2) if self.checkpoints[i] is not None  or i == 2]
        for i in range(n_threads):
            # checkpointIndex = random.choice(activeCheckpoints)
            checkpointIndex = max(2, random.choices(np.arange(boardSize**2), weights=self.transitionPerProgress)[0] - checkpointSetback)
            if random.random() < min(advanceProb, 1 - (self.furthestCheckpoint / boardSize**2)):
                checkpointIndex = max(2, random.randint(self.furthestCheckpoint-10, self.furthestCheckpoint))
            
            while self.checkpoints[checkpointIndex] is None and checkpointIndex > 2:
                checkpointIndex -= 1
            
            if checkpointIndex == 2:
                env = Environment()
            else:
                env = self.checkpoints[checkpointIndex]
            trajectories.append([{
                'env': env,
                'reward': None,
                'emp_val': None,
                'evaluation': None,
                'GAE': None
            }])
        value_hist = []
        action_hist = []
        log_prob_hist = []
        dist_hist = []
        for t in range(maxEpisodeTime):
            inputs = []
            for i in active_thread_hist[t]:
                inputs.append(trajectories[i][t]['env'].toTensor())
            logits, value = self.model(torch.stack(inputs).to(device))

            assert (~logits.isnan()).all()

            valid_acts = []
            for index, i in enumerate(active_thread_hist[t]):
                va = trajectories[i][t]['env'].validActions()
                if va.sum() == 0:
                    print(trajectories[i][t]['env'])
                assert va.sum() > 0
                valid_acts.append(va)
            valid_acts = torch.stack(valid_acts).to(device)

            logits = logits.masked_fill(~valid_acts, float('-inf'))
            
            dist = Categorical(logits=logits)

            actions = dist.sample()

            # I shouldn't have to do this but somehow I do:
            for resample_it in range(10):
                if (valid_acts[torch.arange(len(actions)), actions]).all():
                    break
                actions = dist.sample()

            assert (valid_acts[torch.arange(len(actions)), actions]).all()

            log_prob = dist.log_prob(actions)
            new_active_threads = copy.deepcopy(active_thread_hist[t])
            for index, i in enumerate(active_thread_hist[t]):
                new_env = copy.deepcopy(trajectories[i][t]['env'])
                trajectories[i][t]['evaluation'] = value[index].item()
                trajectories[i][t]['reward'], endState = new_env.makeAction(actions[index].item())
                if endState:
                    new_active_threads.remove(i)
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
        
        # Get terminal values
        if len(active_thread_hist) == maxEpisodeTime+1:
            inputs = []
            for i in active_thread_hist[maxEpisodeTime]:
                assert len(trajectories[i][maxEpisodeTime]['env'].validDirs()) > 0
                inputs.append(trajectories[i][maxEpisodeTime]['env'].toTensor())
            logits, value = self.model(torch.stack(inputs).to(device))
            for index, i in enumerate(active_thread_hist[maxEpisodeTime]):
                trajectories[i][maxEpisodeTime]['evaluation'] = value[index].item()

        for i in range(n_threads):
            gae_val = 0
            emp_val = trajectories[i][-1]['evaluation']
            for t in range(len(trajectories[i])-2, -1, -1):
                time_diff = trajectories[i][t+1]['env'].time - trajectories[i][t]['env'].time
                gae_val = (trajectories[i][t]['reward'] 
                       + discountRate**time_diff * trajectories[i][t+1]['evaluation'] 
                       - trajectories[i][t]['evaluation']
                       + (discountRate * GAEparam) ** time_diff * gae_val)
                trajectories[i][t]['GAE'] = gae_val
                emp_val = trajectories[i][t]['reward'] + discountRate**time_diff * emp_val
                trajectories[i][t]['emp_val'] = emp_val
        return trajectories, active_thread_hist, value_hist, action_hist, dist_hist
    
    def evaluationRollouts(self, n_threads):
        # Doesn't store trajectory
        self.model.eval()
        envs = []
        active_threads = list(range(n_threads))
        for i in range(n_threads):
            envs.append(Environment())
        
        rewardSums = [0] * n_threads
        lengths = [0] * n_threads
        
        for t in range(maxRolloutTime):

            inputs = []
            valid_acts = []
            for i in active_threads:
                inputs.append(envs[i].toTensor())
                valid_acts.append(envs[i].validActions())
            
            logits, value = self.model(torch.stack(inputs).to(device))
            valid_acts = torch.stack(valid_acts).to(device)


            logits = logits.masked_fill(~valid_acts, float('-inf'))
            
            dist = Categorical(logits=logits)

            actions = dist.sample()

            for resample_it in range(10):
                if (valid_acts[torch.arange(len(actions)), actions]).all():
                    break
                actions = dist.sample()
            
            assert (valid_acts[torch.arange(len(actions)), actions]).all()

            next_active_threads = copy.deepcopy(active_threads)
            
            for index, i in enumerate(active_threads):
                reward, endState = envs[i].makeAction(actions[index].item())
                if endState:
                    next_active_threads.remove(i)
                lengths[i] += 1
                rewardSums[i] += reward
            
            active_threads = next_active_threads

            if len(next_active_threads) == 0:
                break
        return envs, rewardSums, lengths
    
    def PPOupdate(self, batchSize, trajectories, active_thread_hist, action_hist, dist_hist):
        self.model.train()
        identifiers = []
        for t in range(len(active_thread_hist)):
            for index in range(len(active_thread_hist[t])):
                if t < maxEpisodeTime:
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

                assert (~old_logprob.isnan()).all()

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)

                assert (valid_acts.sum(axis=1) > 0).all()
                assert (~logits.isnan()).all()

                logits = logits.masked_fill(~valid_acts, float('-inf'))

                curr_dist = Categorical(logits=logits)

                assert (valid_acts[torch.arange(len(actions)), actions]).all()

                ratio = torch.exp(curr_dist.log_prob(actions) - old_logprob)
                # ratio = curr_dist.probs[torch.arange(len(actions)), actions] / np.exp(old_logprob)
                clipped = ratio.clip(max=1+PPOeps, min=1-PPOeps)
                CLIP_loss = -torch.min(ratio * advantages, clipped * advantages)

                loss = (CLIP_loss
                         + value_coef * (value.reshape((-1,)) - emp_vals) ** 2 
                         - self.entropy_coef * curr_dist.entropy()).mean()
                loss.backward()
                self.optimizer.step()

                inputs = []
                emp_vals = []
                advantages = []
                actions = []
                old_logprob = []
                valid_acts = []


    def trainLoop(self, n_iter, n_threads, evalPeriod, rolloutPeriod, gameLogPeriod, include_accuracy=False):

        scoreHist = []
        lengthsHist = []
        timesHist = []
        winHist = []
        lossHist = []
        progressHist = []

        printLineToFile(self.mainOutput, "Starting training, Previous iterations: " + str(self.itCount) + " Current iterations: " + str(n_iter) + " Timestamp: " + str(datetime.now()))
        printLineToFile(self.mainOutput, f'Environment: {environment_name}, boardSize: {boardSize}, maxEpisodeTime: {maxEpisodeTime}, appleReward: {appleReward}, loseReward: {loseReward}, winReward: {winReward}, actionReward: {actionReward}, discountRate: {discountRate}, GAEParam: {GAEparam}')
        printLineToFile(self.mainOutput, f'Value coef: {value_coef}, Entropy coef: {entropy_coef}, Learning rate: {learning_rate}, PPOeps: {PPOeps}, batchSize: {batchSize}, numThreads: {n_threads}, transitionDecay: {transitionDecay}, checkpointSetback: {checkpointSetback}, numReplacements: {numReplacements}, advanceProb: {advanceProb}')

        for it in range(n_iter):
            
            # with open(self.debugOutput, 'a') as f:
                # s = 0
                # c = 0
                # for param_tens in self.model.parameters():
                #     s += param_tens.abs().sum()
                #     c += len(param_tens.flatten())
                # f.write(f"Iteration {it}, Abs net weight: {s/c}\n")

            trajectories, active_thread_hist, value_hist, action_hist, dist_hist = self.generateTrajectories(n_threads)
            self.PPOupdate(batchSize, trajectories, active_thread_hist, action_hist, dist_hist)

            # Update checkpoints

            for traj in trajectories:
                prevSize = None
                numSteps = 0
                cum_time = 0
                for t in range(len(traj)-1):
                    size = (traj[t]['env'].body != -1).sum() + 1
                    if self.checkpoints[size] is None or random.random() < numReplacements / (n_threads * maxEpisodeTime):
                        self.checkpoints[size] = traj[t]['env']
                    if prevSize != None and size > prevSize:
                        self.transitionLengths[prevSize] = (transitionDecay * self.transitionLengths[prevSize] + 
                                                      (1 - transitionDecay) * numSteps)
                        self.transitionPerProgress[prevSize] = (transitionDecay * self.transitionPerProgress[prevSize] + 
                                                      (1 - transitionDecay) * cum_time / (size - prevSize))
                        numSteps = 0
                        cum_time = 0
                    prevSize = size
                    numSteps += 1
                    cum_time += traj[t+1]['env'].time - traj[t]['env'].time
            for i in range(boardSize**2):
                if self.checkpoints[i] is not None:
                    self.furthestCheckpoint = i

            # Evaluation

            scores = []
            lengths = []
            sizeDiffs = []
            times = []
            wins = []
            losses = []

            for traj in trajectories:
                score = sum(s['reward'] for s in traj[:-1])
                scores.append(score)

                game_length = len(traj)-1
                lengths.append(game_length)

                elapsed_time = traj[-1]['env'].time - traj[0]['env'].time
                times.append(elapsed_time)

                sizeDiff = (traj[-1]['env'].body != -1).sum() - (traj[0]['env'].body != -1).sum()
                sizeDiffs.append(sizeDiff)

                win = (traj[-1]['env'].body == -1).sum() == 1
                wins.append(win)
                losses.append((len(traj[-1]['env'].validDirs()) == 0) and not win)
                        
            scoreHist.append(np.array(scores).mean())
            timesHist.append(np.array(times).mean())
            lengthsHist.append(np.array(lengths).mean())
            winHist.append(np.array(wins).mean())
            lossHist.append(np.array(losses).mean())
            progressHist.append(np.array(sizeDiffs).sum() / np.array(times).sum())
            
            
            

            self.itCount += 1
            if self.itCount % evalPeriod == 0:
                winRate = np.array(winHist[-evalPeriod:]).mean()
                printLineToFile(self.mainOutput, "Iteration: " + str(self.itCount) + 
                            " Score: " + colored(str(np.array(scoreHist[-evalPeriod:]).mean()), 'red') + 
                            " Game Length: " + str(np.array(lengthsHist[-evalPeriod:]).mean()) + 
                            " Elapsed Time: " + str(np.array(timesHist[-evalPeriod:]).mean()) + 
                            " Wins: " + str(winRate) + 
                            " Losses: " + str(np.array(lossHist[-evalPeriod:]).mean()) + 
                            " Furthest Checkpoint: " + str(self.furthestCheckpoint) + 
                            " Progress: " + colored(str(np.array(progressHist[-evalPeriod:]).mean()), 'green') + 
                            " Timestamp: " + str(datetime.now()))
                
                if self.itCount % rolloutPeriod == 0:
                    envs, rewards, lengths = self.evaluationRollouts(n_threads)
                    wins = []
                    losses = []
                    winTime = []
                    times = []
                    sizes = []
                    for env in envs:
                        size = (env.body != -1).sum() + 1
                        win = size == boardSize ** 2
                        sizes.append(size)
                        wins.append(win)
                        loss = len(env.validDirs()) == 0 and not win
                        losses.append(loss)
                        times.append(env.time)
                        if win:
                            winTime.append(env.time)
                    evalWinRate = np.array(wins).mean()
                    evalLossRate = np.array(losses).mean()
                    evalWinTime = "NA" if evalWinRate == 0 else np.array(winTime).sum() / np.array(wins).sum()
                    evalSize = np.array(sizes).mean()
                    evalTime = np.array(times).mean()

                    printLineToFile(self.mainOutput, "    " + 
                                    " Eval score: " + str(np.array(rewards).mean()) + 
                                    " Eval length: " + str(np.array(lengths).mean()) + 
                                    " Eval size: " + str(evalSize) + 
                                    " Eval wins: " + colored(str(evalWinRate), 'blue') + 
                                    " Eval losses: " + str(evalLossRate) + 
                                    " Eval time: " + str(evalTime) + 
                                    " Eval win time: " + colored(str(evalWinTime), 'blue'))

                if winRate > anneal_winRate and not self.annealed:
                    self.annealed = True
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=anneal_learning_rate)
                    self.entropy_coef = anneal_entropy_coef
                    printLineToFile(self.mainOutput, f"Iteration {self.itCount}: Learning rate set to {anneal_learning_rate}. Entropy coef set to {self.entropy_coef}.")
                
                if self.itCount % gameLogPeriod == 0:
                    with open(self.gameOutput, 'a') as f:
                        s = "----------------------------- Iteration " + str(self.itCount) + ' -----------------------------\n'
                        index = 0
                        # for index in range(n_threads):
                        s += '"################ Thread ' + str(index) + ' ################\n'
                        for t in range(min(len(active_thread_hist), maxEpisodeTime)):
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

                with open(self.debugOutput, 'a') as f:
                    f.write("Iteration " + str(it) + ' Lengths: ' + str(self.transitionLengths) + '\n')
                    f.write('    Time per progress: ' + str(self.transitionPerProgress) + '\n')

mode = sys.argv[1]

if mode == 'WRITE':
    ppo = PPO()
else:
    assert mode == 'READ'
    with open('snake.pkl', 'rb') as file:
        ppo = pickle.load(file)

ppo.trainLoop(n_iter=int(sys.argv[2]),
             n_threads=int(sys.argv[3]), 
             evalPeriod=int(sys.argv[4]),
             rolloutPeriod=int(sys.argv[5]),
             gameLogPeriod=int(sys.argv[6]))