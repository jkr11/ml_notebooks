# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layer: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
NUM_EPOCHS = 5
BS = 128

if __name__ == "__main__": 
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data/", train=True, download=True, transform=transform)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=BS,
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.MNIST("data/", train=False, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    model = MLP(input_size=28*28, hidden_layer=64, output_size=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if(i % 99 == 0): 
                print(f'loss at step {i}: { loss}')
            # rescale the loss to be a mean over the global batch size instead of
            optimizer.zero_grad()
            # compute the gradients locally
            loss.backward()
            optimizer.step()
        
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the model on the {} test images: {} %".format(
                total, 100 * correct / total
            ),
        )


#Relu: 95.97% GeLu: 96.08%  Tanh + gelu 96.55%  tanh 96.67%


