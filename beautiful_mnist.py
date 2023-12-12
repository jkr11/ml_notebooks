import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
          nn.Conv2d(1, 32, 5), nn.ReLU(),
          nn.Conv2d(32, 32, 5), nn.ReLU(),
          nn.BatchNorm2d(32), nn.MaxPool2d(2),
          nn.Conv2d(32, 64, 3), nn.ReLU(),
          nn.Conv2d(64, 64, 3), nn.ReLU(),
          nn.BatchNorm2d(64), nn.MaxPool2d(2),
          nn.Flatten(), nn.Linear(576, 10)
        )

    def forward(self, x):
        return self.layers(x)

def get_test_acc():
    with torch.no_grad():
        model.eval()
        pred = model(X_test.unsqueeze(1).float()).argmax(dim=1)
        return (pred == y_test).float().mean().item() * 100

test_acc = float('nan')

if __name__ == "__main__":
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    X_train, y_train, X_test, y_test = mnist.train_data, mnist.train_labels, mnist.test_data, mnist.test_labels

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = Model().to(device)
    opt = optim.Adam(model.parameters())
    
    def train_step(batch_indices):
        opt.zero_grad()
        input_data = X_train[batch_indices].unsqueeze(1).float()
        labels = y_train[batch_indices]
        loss = nn.CrossEntropyLoss()(model(input_data), labels)
        loss.backward()
        opt.step()
        return loss.item()
        
    for i in (t:=trange(70)):
        batch_indices = torch.randint(0, len(X_train), (512,))
        loss = train_step(batch_indices)
        if i%10 == 9: test_acc = get_test_acc()
        t.set_description(f"loss: {loss:6.2f} test_accuracy: {test_acc:5.2f}%")