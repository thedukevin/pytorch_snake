
'''
Again using:

https://cp-algorithms.com/graph/cutpoints.html
'''


import numpy as np
import copy

np.random.seed(42)

boardSize = 30

dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def shiftPos(pos, d):
    return (pos[0] + dir[d][0], pos[1] + dir[d][1])

def isValid(pos):
    return 0 <= pos[0] < boardSize and 0 <= pos[1] < boardSize

class Environment:
    def __init__(self):
        self.body = np.full((boardSize, boardSize), -1)
        self.head = (boardSize//2, 2)
        self.tail = (boardSize//2, 1)
        self.body[self.tail] = 0
        self.randomizeApple()
        self.time = 0
    
    def validDirs(self):
        validDirs = []
        for d in range(4):
            newHead = shiftPos(self.head, d)
            if isValid(newHead) and self.body[newHead] == -1:
                validDirs.append(d)
        return validDirs

    def detAction(self, d): # returns whether apple was acheived
        assert d in self.validDirs()
        self.body[self.head] = d
        self.head = shiftPos(self.head, d)
        self.time += 1

        if self.head == self.apple:
            return 1
        else:
            tailDir = self.body[self.tail]
            self.body[self.tail] = -1
            self.tail = shiftPos(self.tail, tailDir)
            return 0
    
    
    def randomizeApple(self):
        openSquares = [(i, j) for i in range(boardSize) for j in range(boardSize) if self.body[i][j] == -1 and (i, j) != self.head]
        if len(openSquares) == 0:
            return
        self.apple = openSquares[np.random.randint(len(openSquares))]
    
    def __str__(self):
        s = "Time: " + str(self.time) + '\n'
        s += "Head: " + str(self.head) + '\n'
        s += "Apple: " + str(self.apple) + '\n'
        s += "Size: " + str((self.body != -1).sum() + 1) + '\n'
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

def heuristic(env):
    # print(env)
    # Get dynamic distance

    visited = np.zeros((boardSize, boardSize))
    retractible = env.body.copy()
    curr_tail = env.tail

    queue = [env.head]
    visited[env.head] = 1

    finished = False
    dist = None
    for i in range(boardSize**2):
        next_queue = []
        for pos in queue:
            if pos == env.apple:
                dist = i
                finished = True
                break
            for d in range(4):
                newPos = shiftPos(pos, d)
                if isValid(newPos) and (retractible[newPos] == -1) and (visited[newPos] == 0):
                    visited[newPos] = 1
                    next_queue.append(newPos)
        if finished:
            break
        if curr_tail != env.head:
            retractible[curr_tail] = -1
            for d in range(4):
                neigh = shiftPos(curr_tail, d)
                if isValid(neigh) and visited[neigh]:
                    next_queue.append(neigh)
            curr_tail = shiftPos(curr_tail, env.body[curr_tail])
        queue = next_queue
    
    if env.apple != (-1, -1):
        assert dist is not None
    
    distBonus = -0.02 * dist if dist is not None else 0

    # Get tail vis

    visited = np.zeros((boardSize, boardSize))
    tin = np.full((boardSize, boardSize), -1)
    low = np.full((boardSize, boardSize), -1)
    isAP = np.zeros((boardSize, boardSize))

    tarjan_timer = 0

    def dfs(v, p=None):
        nonlocal tarjan_timer
        visited[v] = 1
        tin[v] = low[v] = tarjan_timer
        tarjan_timer += 1
        children = 0
        for d in range(4):
            newPos = shiftPos(v, d)
            if not isValid(newPos) or env.body[newPos] != -1 or newPos == p:
                continue
            if visited[newPos]:
                low[v] = min(low[v], tin[newPos])
            else:
                dfs(newPos, v)
                low[v] = min(low[v], low[newPos])
                if low[newPos] >= tin[v] and p is not None:
                    isAP[v] = 1
                children += 1
        if p is None and children > 1:
            isAP[v] = 1
        
    dfs(env.head)

    # print(np.where(isAP == 1))

    components = np.full((boardSize, boardSize), -1)
    componentCount = 0

    visibility = {}
    componentSizes = {}
    compConnectedAPs = {}

    def fillComponent(start, compID):
        queue = [start]
        size = 1
        connectedAPs = []
        while len(queue) > 0:
            top = queue.pop()
            for d in range(4):
                newPos = shiftPos(top, d)
                if isValid(newPos) and env.body[newPos] == -1 and components[newPos] == -1:
                    if isAP[newPos] == 0:
                        components[newPos] = compID
                        queue.append(newPos)
                        size += 1
                    else:
                        connectedAPs.append(newPos)
        componentSizes[compID] = size
        return connectedAPs

    queue = [env.head]
    if isAP[env.head]:
        visibility[env.head] = 1
        components[env.head] = -2
    else:
        components[env.head] = componentCount
        compConnectedAPs[componentCount] = fillComponent(env.head, componentCount)
        visibility[componentCount] = componentSizes[componentCount]
        componentCount += 1

    while len(queue) > 0:
        top = queue.pop()
        if isAP[top] == 1:
            for d in range(4):
                newPos = shiftPos(top, d)
                if isValid(newPos) and (env.body[newPos] == -1) and (components[newPos] == -1):
                    queue.append(newPos)
                    if isAP[newPos]:
                        components[newPos] = -2
                        visibility[newPos] = visibility[top] + 1
                    else:
                        components[newPos] = componentCount
                        compConnectedAPs[componentCount] = fillComponent(newPos, componentCount)
                        visibility[componentCount] = visibility[top] + componentSizes[componentCount]
                        componentCount += 1
        else:
            for conn in compConnectedAPs[components[top]]:
                if components[conn] == -1:
                    components[conn] = -2
                    visibility[conn] = visibility[components[top]] + 1
                    queue.append(conn)

    toTailVis = -1
    for d in range(4):
        tail_neigh = shiftPos(env.tail, d)
        if isValid(tail_neigh) and env.body[tail_neigh] == -1 and visited[tail_neigh] == 1:
            if isAP[tail_neigh] == 1:
                vis = visibility[tail_neigh]
            else:
                assert components[tail_neigh] >= 0
                vis = visibility[components[tail_neigh]]
            toTailVis = max(toTailVis, vis)
    
    if toTailVis != -1:
        maxVis = toTailVis
        tailVis = 1
    else:
        maxSize = 0
        maxID = None
        for compID in componentSizes:
            if maxSize < componentSizes[compID]:
                maxSize = componentSizes[compID]
                maxID = compID
        maxVis = visibility[maxID]
        tailVis = 0

    # for first_step in range(4):
    #     start = shiftPos(env.head, first_step)
    #     if not isValid(start) or env.body[start] != -1 or visited[start]:
    #         continue
        
    #     tarjan_timer = 0
    #     def dfs(v, p=None):
    #         global tarjan_timer
    #         visited[v] = 1
    #         tin[v] = low[v] = tarjan_timer
    #         tarjan_timer += 1

    #         size[v] = 1
            
    #         for d in range(4):
    #             newPos = shiftPos(v, d)
    #             if isValid(newPos) and (env.body[newPos] == -1) and newPos != env.head:
    #                 if newPos == p:
    #                     continue
    #                 if visited[newPos]:
    #                     low[v] = min(low[v], tin[newPos])
    #                 else:
    #                     dfs(newPos, v)
    #                     low[v] = min(low[v], low[newPos])
    #                     if low[newPos] >= tin[v] and p is not None:
    #                         isAP[v] = 1
        
    #     dfs(start)

    # visited = np.zeros((boardSize, boardSize))
    # maxVis = 0
    # currVis = 0
    # tailVis = False
    # for first_step in range(4):
    #     start = shiftPos(env.head, first_step)
    #     if not isValid(start) or env.body[start] != -1 or visited[start]:
    #         continue
    #     queue = [start]
    #     visited[start] = 1
    #     while len(queue) > 0:
    #         top = queue.pop()
    #         for d in range(4):
    #             newPos = shiftPos(top, d)
    #             if isValid(newPos) and (env.body[newPos] == -1) and (visited[newPos] == 0) and newPos != env.head:
    #                 visited[newPos] = 1
    #                 queue.append(newPos)
    #             if newPos == env.tail:
    #                 tailVis = True
    #     maxVis = max(maxVis, visited.sum() - currVis)
    #     currVis = visited.sum()
    
    propVis = maxVis / (env.body == -1).sum()

    tailVisBonus = 0 * tailVis
    propVisBonus = propVis # + min((propVis-0.1) * 50, 0)
    
    return distBonus + tailVisBonus + propVisBonus, (dist, tailVis, propVis)

    # return -0.03 * (abs(env.head[0] - env.apple[0]) + abs(env.head[1] - env.apple[1]))

def printToFile(file, text):
    if file is None:
        print(text)
    else:
        with open(file, 'a') as f:
            f.write(text + '\n')

def sim():
    env = Environment()

    # outputFile = 'snake.txt'
    outputFile = None

    if outputFile is not None:
        with open(outputFile, 'w') as f:
            f.write("")

    for t in range(50000):
        vals = []
        features = []
        for d in range(4):
            if d not in env.validDirs():
                vals.append(-10**9)
                features.append(None)
                continue
            curr_env = copy.deepcopy(env)
            _ = curr_env.detAction(d)
            heuris_val, f = heuristic(curr_env)
            vals.append(heuris_val)
            features.append(f)
        
        sorted_actions = [(vals[i], i) for i in range(4)]
        sorted_actions.sort(reverse=True)

        safeAct = None

        for val, act in sorted_actions:
            if act not in env.validDirs():
                continue
            traj = copy.deepcopy(env)
            actionHist = [act]
            traj.detAction(act)
            safe = False
            for i in range(300):
                _, (dist, tailVis, propVis) = heuristic(traj)
                if tailVis:
                    safe = True
                    break
                
                bestVal = -10**9
                bestAct = None
                for d in traj.validDirs():
                    curr_env = copy.deepcopy(traj)
                    curr_env.detAction(d)
                    heuris_val, f = heuristic(curr_env)
                    if heuris_val > bestVal:
                        bestVal = heuris_val
                        bestAct = d
                traj.detAction(bestAct)
                actionHist.append(bestAct)
                if traj.head == traj.apple:
                    traj.apple = (-1, -1)
                if len(traj.validDirs()) == 0:
                    break
            if safe:
                safeAct = act
                break
        
        if safeAct is None:
            for act in safeActions:
                env.detAction(act)
        else:
            safeActions = actionHist
            for i in range(len(safeActions)):
                env.detAction(safeActions[0])
                safeActions.pop(0)
                if env.head == env.apple:
                    env.randomizeApple()
                    break

        printToFile(outputFile, str(env))
        printToFile(outputFile, str(vals))
        printToFile(outputFile, str(features))

        if len(env.validDirs()) == 0:
            break
    return (env.body != -1).sum() + 1, env.time

# def sim():
#     env = Environment()

#     outputFile = 'snake.txt'

#     with open(outputFile, 'w') as f:
#         f.write("")

#     for t in range(50000):
#         bestValue = -10**9
#         bestEnv = None
#         vals = []
#         features = []
#         bestFeatures = None
#         for d in range(4):
#             curr_env = copy.deepcopy(env)
#             for i in range(1):
#             # for i in range(boardSize):
#                 if d not in curr_env.validDirs():
#                     vals.append(None)
#                     features.append(None)
#                     break
#                 apple = curr_env.detAction(d)
#                 heuris_val, f = heuristic(curr_env)
#                 value = apple - i * stepPenalty + heuris_val
#                 vals.append(value)
#                 features.append(f)
#                 if bestValue < value:
#                     bestValue = value
#                     bestFeatures = f
#                     bestEnv = copy.deepcopy(curr_env)

#         dist, tailVis, propVis = bestFeatures
#         # if not tailVis:
#         #     break
#         with open(outputFile, 'a') as f:
#             f.write(str(env) + '\n')
#             f.write(str(vals) + '\n')
#             f.write(str(features) + '\n')

#         env = bestEnv
#         if env.head == env.apple:
#             env.randomizeApple()
#         while len(env.validDirs()) == 1:
#             env.detAction(env.validDirs()[0])
#             if env.head == env.apple:
#                 env.randomizeApple()
#         if len(env.validDirs()) == 0:
#             break
#     return (env.body != -1).sum() + 1, env.time

sizes = []
lengths = []
wins = []

for i in range(1):
    size, length = sim()
    print(size, length)
    sizes.append(size)
    lengths.append(length)
    win = size == boardSize ** 2
    wins.append(win)

# print("Average size: " + str(np.array(sizes).mean()))
# print("Average length: " + str(np.array(lengths).mean()))
# print("Average wins: " + str(np.array(wins).mean()))