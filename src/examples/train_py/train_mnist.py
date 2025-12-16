import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5

# Data loading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}"
            )


# Testing
def test():
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy


def save_checkpoint(epoch, loss, accuracy):
    ckpt_dir = Path("checkpoints/mnist")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights in SafeTensors format
    weights_path = ckpt_dir / f"epoch_{epoch}.safetensors"
    save_file(model.state_dict(), weights_path)

    # Save metadata separately as JSON
    metadata = {"epoch": epoch, "loss": loss, "accuracy": accuracy}
    meta_path = ckpt_dir / f"epoch_{epoch}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"Checkpoint saved: {weights_path}")


def load_checkpoint(epoch):
    model = MLP()
    ckpt_dir = Path("checkpoints/mnist")
    model.load_state_dict(load_file(ckpt_dir / f"epoch_{epoch}.safetensors"))
    return model


# Run
if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        loss, accuracy = test()
    save_checkpoint(epoch, loss, accuracy)
