
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

N = 5

n_inputs = N*2
n_outputs = N+1

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(n_inputs, 20)
        self.L2 = nn.Linear(20, 20)
        self.L3 = nn.Linear(20, n_outputs)
    
    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.L3(x)
        return x

# Define the problem

datasetSize = 1 << (2 * N)

X = []
y = []

def convertBinary(x, numDigits):
    s = bin(x)[2:]
    return [0] * (numDigits - len(s)) + [int(c) for c in s]

for i in range(1 << N):
    for j in range(1 << N):
        X.append(convertBinary(i, N) + convertBinary(j, N))
        y.append(convertBinary(i + j, N + 1))

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Train the model

batchSize = 32

model = MLP()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)

hist = []
accuracy = []

numIter = 5000
evalPeriod = 100

for i in range(numIter):
    if i % evalPeriod == 0:
        print("Running iteration " + str(i))
        accuracy.append(((model(X) > 0.5) == y).all(axis=1).sum() / datasetSize)
    optimizer.zero_grad()

    batchIndex = torch.randint(high=datasetSize, size=(batchSize,))

    outputs = model(X[batchIndex])
    loss = criterion(outputs, y[batchIndex])
    loss.backward()
    optimizer.step()

    hist.append(loss.item())

hist = torch.tensor(hist)
accuracy = torch.tensor(accuracy)

plt.plot(hist.reshape((numIter // evalPeriod, evalPeriod)).mean(axis=1))
plt.plot(accuracy)
plt.savefig('loss')