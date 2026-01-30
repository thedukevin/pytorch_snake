
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

def manhattanDist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

    def detAction(self, d, forceGrow=False): # returns whether apple was acheived
        assert d in self.validDirs()
        self.body[self.head] = d
        self.head = shiftPos(self.head, d)
        self.time += 1

        if self.head == self.apple:
            return 1
        elif not forceGrow:
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

    dist = None
    if env.apple != (-1, -1):

        queue = [env.head]
        visited[env.head] = 1

        finished = False
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
    
        assert dist is not None
    
    distBonus = -dist if dist is not None else 0

    # Get visibility after retraction - i.e. using retractible as graph

    retractible[curr_tail] = -1

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
            if not isValid(newPos) or retractible[newPos] != -1 or newPos == p:
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

    componentSizes = {}
    compConnectedAPs = {}
    parent = {}

    def fillComponent(start, compID):
        queue = [start]
        size = 1
        connectedAPs = []
        while len(queue) > 0:
            top = queue.pop()
            for d in range(4):
                newPos = shiftPos(top, d)
                if isValid(newPos) and retractible[newPos] == -1 and components[newPos] == -1:
                    if isAP[newPos] == 0:
                        components[newPos] = compID
                        queue.append(newPos)
                        size += 1
                    else:
                        connectedAPs.append(newPos)
        componentSizes[compID] = size
        return connectedAPs

    # Find visibility from apple position

    if env.apple == (-1, -1):
        start = env.head
    else:
        start = env.apple

    queue = [start]
    if isAP[start]:
        components[start] = -2
    else:
        components[start] = componentCount
        compConnectedAPs[componentCount] = fillComponent(start, componentCount)
        componentCount += 1

    while len(queue) > 0:
        top = queue.pop()
        if isAP[top] == 1:
            for d in range(4):
                newPos = shiftPos(top, d)
                if isValid(newPos) and (retractible[newPos] == -1) and (components[newPos] == -1):
                    queue.append(newPos)
                    if isAP[newPos]:
                        components[newPos] = -2
                        parent[newPos] = top
                    else:
                        components[newPos] = componentCount
                        compConnectedAPs[componentCount] = fillComponent(newPos, componentCount)
                        parent[componentCount] = top
                        componentCount += 1
        else:
            for conn in compConnectedAPs[components[top]]:
                if components[conn] == -1:
                    components[conn] = -2
                    parent[conn] = int(components[top])
                    queue.append(conn)
    
    for i in range(boardSize):
        for j in range(boardSize):
            if retractible[i, j] == -1 and components[i, j] == -1:
                components[i, j] = componentCount
                _ = fillComponent((i, j), componentCount)
                componentCount += 1

    headComp = env.head if isAP[env.head] else int(components[env.head])
    tailComp = curr_tail if isAP[curr_tail] else int(components[curr_tail])

    def addVis(comp, rootComp):
        while comp in parent and comp != rootComp:
            visibleComponents.add(comp)
            comp = parent[comp]
    
    if env.apple != (-1, -1):
        appleComp = env.apple if isAP[env.apple] else int(components[env.apple])
        visibleComponents = {appleComp}

        addVis(headComp, appleComp)
        addVis(tailComp, appleComp)

        # trace = headComp
        # while trace != appleComp:
        #     visibleComponents.add(trace)
        #     trace = parent[trace]
        # trace = tailComp
        # while trace in parent and trace != appleComp:
        #     visibleComponents.add(trace)
        #     trace = parent[trace]
    else:
        visibleComponents = {headComp}

        maxSize = 0
        comp = None
        for compID in componentSizes:
            if maxSize < componentSizes[compID]:
                maxSize = componentSizes[compID]
                comp = compID
        
        addVis(comp, headComp)
        addVis(tailComp, headComp)
    
    sumDist = 0

    for t in range(boardSize**2):
        if curr_tail == env.head:
            break
        curr_tail = shiftPos(curr_tail, env.body[curr_tail])
        for d in range(4):
            tail_neigh = shiftPos(curr_tail, d)
            if isValid(tail_neigh) and retractible[tail_neigh] == -1:
                comp = tail_neigh if isAP[tail_neigh] else components[tail_neigh]
                size = 1 if isAP[tail_neigh] else componentSizes[components[tail_neigh]]
                if comp not in visibleComponents:
                    visibleComponents.add(comp)
                    sumDist += size * max(0, t - manhattanDist(tail_neigh, start))

    avgWaitBonus = -sumDist / (env.body == -1).sum() * ((env.body != -1).sum() / 50)

    # toTailVis = -1
    # for d in range(4):
    #     tail_neigh = shiftPos(env.tail, d)
    #     if isValid(tail_neigh) and env.body[tail_neigh] == -1 and visited[tail_neigh] == 1:
    #         if isAP[tail_neigh] == 1:
    #             vis = visibility[tail_neigh]
    #         else:
    #             assert components[tail_neigh] >= 0
    #             vis = visibility[components[tail_neigh]]
    #         toTailVis = max(toTailVis, vis)
    
    # if toTailVis != -1:
    #     maxVis = toTailVis
    # else:
    #     maxSize = 0
    #     maxID = None
    #     for compID in componentSizes:
    #         if maxSize < componentSizes[compID]:
    #             maxSize = componentSizes[compID]
    #             maxID = compID
    #     maxVis = visibility[maxID]

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

    visited = np.zeros((boardSize, boardSize))
    tailVis = False
    queue = [env.head]
    visited[env.head] = 1
    while len(queue) > 0:
        top = queue.pop()
        for d in range(4):
            newPos = shiftPos(top, d)
            if isValid(newPos) and (env.body[newPos] == -1) and (visited[newPos] == 0):
                visited[newPos] = 1
                queue.append(newPos)
            if top != env.head and newPos == env.tail:
                tailVis = True
    
    # propVis = maxVis / (env.body == -1).sum()

    # tailVisBonus = 0 * tailVis
    # avgW = avgWait # + min((propVis-0.1) * 50, 0)

    isBorder = env.head[0] in (0, boardSize-1) or env.head[1] in (0, boardSize-1)
    # for d in range(4):
    #     head_neigh = shiftPos(env.head, d)
    #     if isValid(head_neigh):
    #         if env.body[head_neigh] != -1 and shiftPos(head_neigh, env.body[head_neigh]) != env.head:
    #             isBorder = True

    borderBonus = 0.5 * isBorder
    
    return distBonus + avgWaitBonus + borderBonus, (dist, tailVis, avgWaitBonus, isBorder)

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
        action_pairs = []
        for d1 in range(4):
            if d1 not in env.validDirs():
                vals.append(None)
                features.append(None)
                continue
            env1 = copy.deepcopy(env)
            isApple = env1.detAction(d1)
            if isApple:
                heuris_val, f = heuristic(env1)
                vals.append(heuris_val+1)
                features.append(f)
                action_pairs.append(((heuris_val, f[1]), (d1, None)))
                continue

            val_row = []
            feat_row = []
            for d2 in range(4):
                if d2 not in env1.validDirs():
                    val_row.append(None)
                    feat_row.append(None)
                    continue
                
                env2 = copy.deepcopy(env1)
                _ = env2.detAction(d2)
                heuris_val, f = heuristic(env2)
                val_row.append(heuris_val)
                feat_row.append(f)
                action_pairs.append(((heuris_val, f[1]), (d1, d2)))
            
            vals.append(val_row)
            features.append(feat_row)
        
        printToFile(outputFile, str(env))
        printToFile(outputFile, str(vals))
        printToFile(outputFile, str(features))

        action_pairs.sort(reverse=True)

        safeAct = None

        for val, (act1, act2) in action_pairs:
            # print(f"Trying: {act1} {act2}")
            traj = copy.deepcopy(env)
            actionHist = [act1]
            traj.detAction(act1)
            if act2 is not None:
                actionHist.append(act2)
                traj.detAction(act2)
            safe = False
            for i in range(300):
                if len(traj.validDirs()) == 0:
                    break
                _, f = heuristic(traj)
                if f[1]:
                    # print("Found safe: ")
                    # print(traj)
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
                traj.detAction(bestAct, forceGrow=(i<2))
                actionHist.append(bestAct)
                if traj.head == traj.apple:
                    traj.apple = (-1, -1)
            if safe:
                safeAct = act1
                break
        
        if safeAct is None:
            assert len(safeActions) > 0
            for act in safeActions:
                env.detAction(act)
        else:
            safeActions = actionHist
            if len(actionHist) == 2:
                env.detAction(safeAct, forceGrow = (env.body != -1).sum() < 100 and np.random.uniform() < 0.3)
                safeActions.pop(0)
            else:
                for i in range(len(safeActions)):
                    env.detAction(safeActions[0], forceGrow = (env.body != -1).sum() < 100 and np.random.uniform() < 0.3)
                    safeActions.pop(0)
                    if env.head == env.apple:
                        env.randomizeApple()
                        break

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