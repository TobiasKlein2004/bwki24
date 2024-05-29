import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights 
from torchvision import transforms
from torchvision.datasets import ImageFolder 
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F

import torch_directml
import time
import glob
import os

from eval import testModel

# Initialize the DirectML device
device_name = torch_directml.device_name(0)
device = torch.device(torch_directml.device(0))




class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.image_paths = []
        for ext in ['png', 'jpg']:
            self.image_paths += glob.glob(os.path.join(root_dir, '*', f'*.{ext}'))
        class_set = set()
        for path in self.image_paths:
            class_set.add(os.path.basename(os.path.dirname(path)))
        self.class_lbl = { cls: i for i, cls in enumerate(sorted(list(class_set)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx], ImageReadMode.RGB).float()
        cls = os.path.basename(os.path.dirname(self.image_paths[idx]))
        label = self.class_lbl[cls]

        img = F.to_pil_image(img /255.)

        return self.transform(img), torch.tensor(label)



def data_loader(data_dir, batch_size):
    # define transforms
    transform = transforms.Compose([
            transforms.Resize((128,128), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
    ])

    # load the dataset
    dataset = CustomDataset(root_dir=data_dir, transform=transform)

    splits = [0.8, 0.1, 0.1] # [train, test, valid]

    split_sizes = []
    for split in splits[:-1]:
        split_sizes.append(int(split*len(dataset)))
    split_sizes.append(len(dataset)-sum(split_sizes))

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, split_sizes)

    val_set.transform = test_set.transform = transforms.Compose([
        transforms.Resize((128, 128))
    ])

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
    }, dataset



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    


# Hyper Parameters  - - - - - - - - - - - - - - - - - 
dataloaders, dataset = data_loader(data_dir='images', batch_size=16)

num_classes = 5
num_epochs = 20
batch_size = 16
learning_rate = 0.01

# model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes).to(device)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, num_classes)
)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  



# Training - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

metrics = {
    'train': {
        'loss': [], 'accuracy': []
    },
    'val': {
        'loss': [], 'accuracy': []
    },
}

lengths = {
    "train": len(dataloaders["train"]),
    "val": len(dataloaders["val"])
}

print("Training Started...")


for epoch in range(num_epochs):
    print(f'\nTraining Epoch {epoch+1}/{num_epochs}...')
    
    start = time.time()

    ep_metrics = {
        'train': {'loss': 0, 'accuracy': 0, 'count': 0},
        'val': {'loss': 0, 'accuracy': 0, 'count': 0},
    }

    for phase in ['train', 'val']:
        for index, (images, labels) in enumerate(dataloaders[phase]):
            print(f'Phase: {phase} -> Batch {index+1}/{lengths[phase]}       ', end="\r")

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'): # Only Enable in Train
                output = model(images.to(device))
                ohe_label = torch.nn.functional.one_hot(labels, num_classes=num_classes)

                loss = criterion(output, ohe_label.float().to(device))

                correct_preds = labels.to(device) == torch.argmax(output, dim=1)
                accuracy = (correct_preds).sum()/len(labels)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            ep_metrics[phase]['loss'] += loss.item()
            ep_metrics[phase]['accuracy'] += accuracy.item()
            ep_metrics[phase]['count'] += 1

        
    ep_loss = ep_metrics[phase]['loss']/ep_metrics[phase]['count']
    ep_accuracy = ep_metrics[phase]['accuracy']/ep_metrics[phase]['count']

    end = time.time()

    print (f'Loss: {ep_loss:.4f}, Accuracy: {round(ep_accuracy*100, 2)}, Time: {round(end-start, 2)}s' )

    metrics[phase]['loss'].append(ep_loss)
    metrics[phase]['accuracy'].append(ep_accuracy)





# Saving: - - - - - - - - - - - - - - - - - - - - - -
print("\nSaving Model...")
torch.save(model, 'model1.pt')
print("Saved!")

testModel(model, dataloaders['test'], num_classes, criterion, dataset)