import torch
import torch.nn as nn
import torch.nn.functional as F

n_input = 3
n_hidden = 4
n_output = 2

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(n_input, n_hidden)
        self.L2 = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.L1(x))
        x = self.L2(x)
        return x

model = MLP()

for name, value in model.named_parameters():
    print(name)
    print(value)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

input = torch.tensor([0., 1., 2.])
label = torch.tensor([-1., 0.])

optimizer.zero_grad()
output = model(input)
loss = criterion(output, label)
loss.backward()
optimizer.step()
