
'''
Again using:

https://cp-algorithms.com/graph/cutpoints.html
'''


import numpy as np
import copy
from termcolor import colored

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
        self.computeBorder()

        # Debugging

        self.components = np.zeros((boardSize, boardSize))
        self.APs = np.zeros((boardSize, boardSize))
    
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

        appleAchieved = 0
        if self.head == self.apple:
            appleAchieved = 1
        elif not forceGrow:
            tailDir = self.body[self.tail]
            self.body[self.tail] = -1
            self.tail = shiftPos(self.tail, tailDir)
        
        self.computeBorder()
        return appleAchieved
    
    def computeBorder(self):

        gridBorder = set((0, i) for i in range(boardSize)) | set((boardSize-1, i) for i in range(boardSize)) | \
            set((i, 0) for i in range(boardSize)) | set((i, boardSize-1) for i in range(boardSize))
        
        touchedBorder = False
        for el in gridBorder:
            if self.body[el] != -1:
                touchedBorder = True
        
        self.border = np.zeros((boardSize, boardSize))

        if not touchedBorder:
            for el in gridBorder:
                self.border[el] = 1
        else:

            for d in range(4):
                head_neigh = shiftPos(self.head, d)
                if isValid(head_neigh) and self.body[head_neigh] != -1 and shiftPos(head_neigh, self.body[head_neigh]) == self.head:
                    blocked1 = head_neigh
            for d in range(4):
                head_neigh = shiftPos(blocked1, d)
                if isValid(head_neigh) and self.body[head_neigh] != -1 and shiftPos(head_neigh, self.body[head_neigh]) == blocked1:
                    blocked2 = head_neigh
            blocked = [self.head, blocked1, blocked2]
            
            def isTouching(pos):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        neigh = (pos[0] + i, pos[1] + j)
                        if not isValid(neigh) or (self.body[neigh] != -1 and neigh not in blocked):
                            return True
                return False
            

            visited = np.zeros((boardSize, boardSize))

            def dfs(start):
                queue = [start]
                visited[start] = 1
                while len(queue) > 0:
                    top = queue.pop()
                    self.border[top] = 1
                    for d in range(4):
                        newPos = shiftPos(top, d)
                        if isValid(newPos) and self.body[newPos] == -1 and isTouching(newPos) and visited[newPos] == 0:
                            visited[newPos] = 1
                            queue.append(newPos)

            for i in range(0, boardSize):
                for j in range(0, boardSize):
                    if self.body[i, j] == -1 and (i, j) not in blocked:
                        dfs((i, j))
                        break
                for j in range(boardSize-1, -1, -1):
                    if self.body[i, j] == -1 and (i, j) not in blocked:
                        dfs((i, j))
                        break
            
            for j in range(0, boardSize):
                for i in range(0, boardSize):
                    if self.body[i, j] == -1 and (i, j) not in blocked:
                        dfs((i, j))
                        break
                for i in range(boardSize-1, -1, -1):
                    if self.body[i, j] == -1 and (i, j) not in blocked:
                        dfs((i, j))
                        break

    
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
                    grid[2*i][2*j] = 'o'
                    grid[2*i + dir[d][0]][2*j + dir[d][1]] = '-' if d%2 == 0 else '|'
                elif (i, j) == self.apple:
                    for d in range(4):
                        if isValid(shiftPos((i, j), d)):
                            grid[2*i+dir[d][0]][2*j+dir[d][1]] = colored('@', "red")
                    grid[2*i][2*j] = colored('@', "red")
                # elif self.border[i, j] == 1:
                #     grid[2*i][2*j] = '*'
                elif self.APs[i, j] == 1:
                    grid[2*i][2*j] = 'A'
                elif self.components[i, j] != 0:
                    grid[2*i][2*j] = chr(self.components[i, j] + 96)
                else:
                    grid[2*i][2*j] = '.'
        s += '\n'.join(''.join(line) for line in grid)
        return s
    
    def toCode(self):
        assert boardSize % 2 == 0
        s = ""
        for i in range(boardSize):
            for j in range(boardSize//2):
                s += chr((self.body[i, 2*j] + 1) * 5 + (self.body[i, 2*j+1] + 1) + 97)
        
        def encodePos(pos):
            return str(pos[0]) + 'x' + str(pos[1])
        return s + '_' + encodePos(self.head) + '_' + encodePos(self.tail) + '_' + encodePos(self.apple) + '_' + str(self.time)
    
    def fromCode(self, code : str):
        body, head, tail, apple, time = code.split('_')
        self.body = np.full((boardSize, boardSize), -1)
        for i in range(boardSize):
            for j in range(boardSize//2):
                c = ord(body[i*boardSize//2+j])-97
                self.body[i, 2*j] = c // 5 - 1
                self.body[i, 2*j+1] = c % 5 - 1
        
        def decodePos(s):
            x, y = s.split('x')
            return int(x), int(y)
        self.head = decodePos(head)
        self.tail = decodePos(tail)
        self.apple = decodePos(apple)
        self.time = int(time)

        self.computeBorder()

def heuristic(env):
    # print(env)

    appleIsBorder = env.border[env.apple]
    headIsBorder = env.border[env.head]

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

    # Get border distance, max it with distBonus.

    if appleIsBorder and headIsBorder:
        visited = np.zeros((boardSize, boardSize))
        borderDist = None
        queue = [env.head]
        visited[env.head] = 1
        finished = False
        for i in range(boardSize**2):
            next_queue = []
            for pos in queue:
                if pos == env.apple:
                    borderDist = i
                    finished = True
                    break
                for d in range(4):
                    newPos = shiftPos(pos, d)
                    if isValid(newPos) and env.border[newPos] == 1 and visited[newPos] == 0:
                        visited[newPos] = 1
                        next_queue.append(newPos)
            if finished or len(queue) == 0:
                break
            queue = next_queue
        if borderDist is not None:
            distBonus = max(distBonus, 50-borderDist)

    # Get visibility

    vis_graph = env.body.copy()

    vis_graph[env.tail] = -1

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
            if not isValid(newPos) or vis_graph[newPos] != -1 or newPos == p:
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
                if isValid(newPos) and vis_graph[newPos] == -1 and components[newPos] == -1:
                    if isAP[newPos] == 0:
                        components[newPos] = compID
                        queue.append(newPos)
                        size += 1
                    else:
                        connectedAPs.append(newPos)
        componentSizes[compID] = size
        return connectedAPs
    
    def searchComponents(start):
        nonlocal componentCount
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
                    if isValid(newPos) and (vis_graph[newPos] == -1) and (components[newPos] == -1):
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
    
    # Find visibility from apple position
    # BUG: apple may now not be visible from head.

    if env.apple == (-1, -1):
        start = env.head
    else:
        start = env.apple
        
    dfs(start)
    searchComponents(start)
    appleVisible = visited[env.head]
    if not appleVisible:
        dfs(env.head)
        searchComponents(env.head)

    for i in range(boardSize):
        for j in range(boardSize):
            if vis_graph[i, j] == -1 and components[i, j] == -1:
                components[i, j] = componentCount
                _ = fillComponent((i, j), componentCount)
                componentCount += 1

    env.APs = isAP
    env.components = components
    # Get visibity from appl

    headComp = env.head if isAP[env.head] else int(components[env.head])
    tailComp = env.tail if isAP[env.tail] else int(components[env.tail])
    appleComp = env.apple if isAP[env.apple] else int(components[env.apple])

    def addVis(comp, rootComp, passThrough=None, addingMode=True):
        ans = False
        addedSize = 0
        while comp in parent and comp != rootComp:
            if passThrough is not None:
                if comp == passThrough and rootComp != passThrough:
                    ans = True
            if addingMode:
                visibleComponents.add(comp)
            addedSize += componentSizes[comp] if comp in componentSizes else 0
            comp = parent[comp]
        return ans, addedSize
    
    split = False
    includeHead = False
    if env.apple != (-1, -1):
        
        visibleComponents = {appleComp}

        split, _ = addVis(tailComp, appleComp, passThrough=headComp)
        if appleVisible:
            fromComp, toComp = headComp, appleComp
        else:
            fromComp, toComp = tailComp, headComp
        _, addedSize = addVis(fromComp, toComp, addingMode=False)
        if addedSize < (env.body == -1).sum() * 0.3:
            includeHead = True
            addVis(fromComp, toComp)

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
    

    # headComp = None
    # if isAP[env.head] == 0:
    #     headComp = components[env.head]
    # else:
    #     for d in range(4):
    #         head_neigh = shiftPos(env.head, d)
    #         if isValid(head_neigh) and retractible[head_neigh] == -1 and isAP[head_neigh] == 0:
    #             headComp = components[head_neigh]
    # if headComp is not None and componentSizes[headComp] < (env.body == -1).sum() * 0.3:
    #     includeHead = True
    #     visibleComponents.add(headComp)
    
    splitBonus = -500 * split
    
    # sumDist = 0
    waits = []
    sizes = []
    waitComps = []

    curr_tail = env.tail

    for t in range(boardSize**2):
        if curr_tail == env.head:
            break
        curr_tail = shiftPos(curr_tail, env.body[curr_tail])
        for d in range(4):
            tail_neigh = shiftPos(curr_tail, d)
            if isValid(tail_neigh) and vis_graph[tail_neigh] == -1:
                comp = tail_neigh if isAP[tail_neigh] else components[tail_neigh]
                size = 1 if isAP[tail_neigh] else componentSizes[components[tail_neigh]]
                if comp not in visibleComponents:
                    visibleComponents.add(comp)
                    wait = max(0, t - manhattanDist(tail_neigh, start))
                    waitComps.append(comp)
                    waits.append(wait)
                    sizes.append(size)
                    # sumDist += size * wait * min(150, wait) / 50
    
    waits.append(0)
    sizes.append((vis_graph == -1).sum() - sum(sizes))

    waits = np.array(waits)
    sizes = np.array(sizes)

    weighted_sizes = (waits + 50) * sizes

    avgWaitBonus = -(waits * weighted_sizes).sum() / weighted_sizes.sum()

    # avgWaitBonus = -sumDist / (env.body == -1).sum()



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
    
    crampedBonus = -5 * (manhattanDist(env.head, env.tail) <= 2)

    # if (env.body != -1).sum() > 700:
    #     return 0, (0, tailVis, "lmao")
    
    return distBonus + avgWaitBonus + splitBonus + crampedBonus, (distBonus, tailVis, avgWaitBonus, appleIsBorder, headIsBorder, splitBonus, includeHead, crampedBonus, waits, sizes, waitComps)

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
    
    progressFile = 'progress.txt'

    with open(progressFile, 'w') as f:
        f.write("")

    actsToApple = 0
    for t in range(50000):
        vals = []
        features = []
        action_vals = []
        for d in range(4):
            if d not in env.validDirs():
                vals.append(None)
                features.append(None)
                continue
            env1 = copy.deepcopy(env)
            isApple = env1.detAction(d)

            heuris_val, f = heuristic(env1)
            vals.append(heuris_val)
            features.append(f)
            action_vals.append(((heuris_val, f[1]), d))
        
        heuristic(env)
        printToFile(outputFile, env.toCode())
        printToFile(outputFile, str(vals))
        printToFile(outputFile, str(features))
        printToFile(outputFile, str(env))

        action_vals.sort(reverse=True)

        safeAct = None

        for val, act in action_vals:
            traj = copy.deepcopy(env)
            actionHist = [act]
            traj.detAction(act)
            safe = False
            for i in range(300):
                if len(traj.validDirs()) == 0:
                    break
                _, f = heuristic(traj)
                if f[1]:
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
                safeAct = act
                break
        
        if safeAct is None:
            assert len(safeActions) > 0
            for act in safeActions:
                env.detAction(act)
                actsToApple += 1
                if env.head == env.apple:
                    with open(progressFile, 'a') as f:
                        f.write(str(actsToApple) + '\n')
                    actsToApple = 0
                    env.randomizeApple()
        else:
            safeActions = actionHist
            for i in range(len(safeActions)):
                env.detAction(safeActions[0])
                actsToApple += 1
                safeActions.pop(0)
                if env.head == env.apple:
                    with open(progressFile, 'a') as f:
                        f.write(str(actsToApple) + '\n')
                    actsToApple = 0
                    env.randomizeApple()
                    break

        if len(env.validDirs()) == 0:
            break
    return (env.body != -1).sum() + 1, env.time

# aaaaaaaaaaaaaaablkaaaaascsncpaeokaaaaaesetsuaeokaaagkaaaaauaeolgggwqghaaauaeooaaabjabhaauaeooaaaaaaacaauaeooaaaaaaabghuaeooaaaaaaansruaeooaaaaaaalhwuaeooaaaaaaaonxuaeooaaaaaaajghuaeooaaaaacsnssuaeooaaaaabohhhuaeooaaaaaaonwwuaeooaaaaaaohwwuaeooaaaaaaonwwuaoooaaaaaaohwwuaoooaaaaactnwwuaoooaaaaacjhwvuaoooaaaaacwswxaaoooaaaaacvhvoaaoooaaaaacwsxjaaoooaaaaacwgjnaaoooaanssrwxstaaooonnlghwwggjaaoooooonswwuaaaaoojoooggwwuaaaaooaottsssxuaaaajjaggggggguaaaa_1x15_17x0_7x19_20935
# nsssssssssnsssslllllllkaeghhgooooooookaesqvwtooooooolkbhxsseooooooookeqggkeooooooookbwsskeooooooookesaekeooooooookaaaakeooooooookaaaakeooooooojlhacskeooooooopjnacekeooooojlwspabokeooooopjbkaaaogooooogwsrpnaaosooojonraxaoaaloooonjkxaactsstooojospacrrhhhjotjnpaansxwvvvwqjnkaadpabwwssqjnokaaaaaenwhhxstokaaaaaegwwwhhjokaaaaaesswwwwxokaaabgghhvwvwjokaaaessrwxnxnxokaaaaaawvololookaaaaaawxjjjjjokaaaaabwjnssssokaaaaaenxggggoopaaaaaelosssstgggggggjjgggggj_18x8_6x25_9x27_40885
# nsssssssaaaaanskbgggkaeaaaaalokennskaeaaaactokeolokaeaaaanjokeoookaeaaaalwtkeooogkepnsstnekeoooskaxlllolokeoologheooooookeoooosqojjjoookeooolhxossrtookeoooonehhhwjookeooooloqvwwwtokeooooohxsrvwjoketjojonhhwxqwtkbopjntlwwvkxnekekwsqoonwwpeqokehvlooolwvllookerxjjooonxjjoooavossooolosrtooaxggojjojlgwjooaensonsjntnswtoaelooqosqjllwjoaeoogjggjntonxoaeoosnsssqoolooaeololgggjojojoaeooojnssrtrtpoaejjospaawhwhxoaessgkaabvvvvjoaaaespaaessaaajaaaaaaaaaaaaaa_17x1_28x23_2x18_45783
# nsssssnssssnssskaaaaelllgjlllokcpaaeooorstoookcxspejolwhjoookcaaxentorwwtookcaaeehjgwwwjookchaeerxpnwwwtokcwaehxhxgwvwjokcwaeqgvorrxnwtlbwaauaakwvjlvoonraaunnhxxrtntohwaaxttqghvoloonwaagghuacxoooohwabunswpbojjoonwaeakaxuctrstoqwaecphbuchwhjloraecawjabwwwxolwaechwuaanvwjoonaecwwuaahxsxoolkecwwuaanghjoojkecwwuaalwsxoonqobvvuantngjoogojaaaaagjlwsoontnsnnsnsstneooljgooljlggjloooonaooosjnsstooottaoogosqlloookgjaoonggjjjjookxsstttsssssstjgggggggggggggj_9x2_7x10_16x7_50808

s = 'aaaaaaaaaaaaaaablkaaaaascsncpaeokaaaaaesetsuaeokaaagkaaaaauaeolgggwqghaaauaeooaaabjabhaauaeooaaaaaaacaauaeooaaaaaaabghuaeooaaaaaaansruaeooaaaaaaalhwuaeooaaaaaaaonxuaeooaaaaaaajghuaeooaaaaacsnssuaeooaaaaabohhhuaeooaaaaaaonwwuaeooaaaaaaohwwuaeooaaaaaaonwwuaoooaaaaaaohwwuaoooaaaaactnwwuaoooaaaaacjhwvuaoooaaaaacwswxaaoooaaaaacvhvoaaoooaaaaacwsxjaaoooaaaaacwgjnaaoooaanssrwxstaaooonnlghwwggjaaoooooonswwuaaaaoojoooggwwuaaaaooaottsssxuaaaajjaggggggguaaaa_1x15_17x0_7x19_20935'

env = Environment()
env.fromCode(s)

print(heuristic(env))
print(env)

# sizes = []
# lengths = []
# wins = []

# for i in range(1):
#     size, length = sim()
#     print(size, length)
#     sizes.append(size)
#     lengths.append(length)
#     win = size == boardSize ** 2
#     wins.append(win)

# print("Average size: " + str(np.array(sizes).mean()))
# print("Average length: " + str(np.array(lengths).mean()))
# print("Average wins: " + str(np.array(wins).mean()))