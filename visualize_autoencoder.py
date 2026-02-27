import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/train", transform=transform)

if len(dataset) == 0:
    raise Exception("Dataset is empty!")

loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = AutoEncoder().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

images, _ = next(iter(loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)

input_image = images.squeeze().cpu().permute(1, 2, 0)
output_image = outputs.squeeze().cpu().permute(1, 2, 0)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(input_image)

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(output_image)

plt.savefig("reconstruction.png")
plt.show()

print("Image saved as reconstruction.png")