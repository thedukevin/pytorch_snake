
from snake import *

with open('snake.pkl', 'rb') as file:
    ppo = pickle.load(file)

def tailVisible(env: Environment):
    visited = np.zeros((boardSize, boardSize))
    queue = [env.head]
    while len(queue) > 0:
        top = queue.pop(0)
        for d in range(4):
            newPos = shiftPos(top, d)
            if isValid(newPos) and env.queryBody(newPos) == -1 and not visited[newPos]:
                queue.append(newPos)
                visited[newPos] = 1
            if newPos == env.tail:
                return True
    return False

def lookAheadRollout():
    env = Environment()
    for t in range(maxRolloutTime):
        input = env.toTensor()
        logits, value = ppo.model(input.reshape(1, 7, 10, 10).to(device))

        logits = logits.masked_fill(~env.validActions(), float('-inf'))

        vals, acts = torch.sort(logits, descending=True)

        action = None
        for a in acts.flatten().tolist():
            if not env.validActions()[a]:
                break
            lookahead = copy.deepcopy(env)
            curr_act = a
            for i in range(10):
                reward, endState = lookahead.makeAction(curr_act)
                if endState:
                    if reward >= -0.5:
                        action = a
                    break
                if tailVisible(lookahead):
                    action = a
                    break
                curr_logits, _ = ppo.model(lookahead.toTensor().reshape(1, 7, 10, 10).to(device))
                curr_logits = curr_logits.masked_fill(~lookahead.validActions(), float('-inf'))
                curr_act = curr_logits.argmax().item()

            if action is not None:
                break
        
        if action is None:
            print(env)
            print(action)
            action = logits.argmax().item()

        reward, endState = env.makeAction(action)

        if endState:
            break
    return t, (env.body != -1).sum() + 1, env.time

wins = 0
winTime = 0

numRollouts = 100

for i in range(numRollouts):
    length, size, t = lookAheadRollout()
    print(i, length, size, t)
    win = size == 100
    wins += win
    winTime += win * t

print("Win Rate: " + str(wins / numRollouts))
print("Win Time: " + str(winTime / wins))
