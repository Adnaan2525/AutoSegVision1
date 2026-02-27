import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset_path = "data/train"

if not os.path.exists(dataset_path):
    raise Exception("Dataset folder not found!")

dataset = datasets.ImageFolder(dataset_path, transform=transform)

if len(dataset) == 0:
    raise Exception("Dataset is empty!")

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")