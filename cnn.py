
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
import copy
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using " + str(device))

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# Datasets & DataLoaders
# -----------------------------
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True
)

# -----------------------------
# Model
# -----------------------------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 7, 7)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTCNN().to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training Loop
# -----------------------------

def trainLoop(epochs):

    print("Training...")

    model_hist = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -----------------------------
        # Evaluation
        # -----------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Test ccuracy: {accuracy:.4f}")
        model_hist.append(copy.deepcopy(model))
    return model_hist

mode = "READ"

if mode == 'WRITE':
    hist = trainLoop(5)
    with open('net.pkl', 'wb') as file:
        pickle.dump(hist, file)
else:
    assert mode == 'READ'
    with open('net.pkl', 'rb') as file:
        hist = pickle.load(file)

summary(hist[0], input_size=(1, 1, 28, 28))

# image, label = train_dataset[0]   # image is a tensor (1, 28, 28)

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(f"Label: {label}")
# plt.axis("off")
# plt.show()

# for m in hist:
#     print(m(image.reshape(1, 1, 28, 28)))

print("Size of test dataset: " + str(len(test_dataset)))

images = []
labels = []

for image, label in test_dataset:
    images.append(image)
    labels.append(label)

predicted = hist[4](torch.stack(images)).argmax(axis=1)

for i in range(len(test_dataset)):
    if labels[i] != predicted[i]:
        plt.figure()
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Label: {labels[i]} Predicted: {predicted[i]}")
        plt.axis("off")
        plt.savefig(f"figs/counter{i}")
        plt.close()