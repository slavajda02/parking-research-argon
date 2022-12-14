import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
from torchvision import datasets
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import os
import time
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Baseline training parameters')

    parser.add_argument('-p', '--path', type=str, help='path to the dataset')

    args = parser.parse_args()

    return args

args = parse_args()

image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

train_directory = f'{args.path}/patch_splitted/train'
valid_directory = f'{args.path}/patch_splitted/val'
test_directory = f'/{args.path}/patch_splitted/test'

bs = 32
num_classes = 2
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
print("Load train data: ", train_data_size)
train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)
print("Load valid data: ", valid_data_size)
valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)
print("Load test data: ", test_data_size)
test_data = DataLoader(data['test'], batch_size=bs, shuffle=True)

class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, dilation=2, padding=(11,10))
        self.conv2 = nn.Conv2d(96, 192, 11, dilation=2, padding=(10, 10))
        self.conv3 = nn.Conv2d(192, 384, 11, dilation=2, padding=(12, 10))
        self.fc1 = nn.Linear(18432, 4096)  # 5*5 from image dimension
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
        self.dropout = nn.Dropout(p=0.4)
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.interpolate(x, size=52)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  
        x = torch.flatten(x, 1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.output(self.fc3(self.dropout(x)))
        return x

model = CarNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type == 'cpu':
    print('Using CPU')
else:
    print('Using GPU')

resnet50 = CarNet().to(device)

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.0001)


model = resnet50

history = []
epochs = 10
for epoch in range(epochs):
    
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    train_f1   = 0.0
    train_ra   = 0.0
    valid_loss = 0.0
    valid_acc  = 0.0
    valid_f1   = 0.0
    valid_ra   = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_data)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)
        
        f1  = f1_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
        train_f1  += f1 * inputs.size(0)
        
        try:
            ra  = roc_auc_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
            train_ra  += ra * inputs.size(0)
        except ValueError:
            print("ROC-AUC ValueError")
            
            print(f"Batch number: {i:03d}, Training: Loss: {train_loss/(i+1)/inputs.size(0):.4f}, Accuracy: {train_acc/(i+1)/inputs.size(0):.4f}, F1: {train_f1/(i+1)/inputs.size(0):.4f}, ROC-AUC: {train_ra/(i+1)/inputs.size(0):.4f}")

    with torch.no_grad():
        model.eval()
        for j, (inputs, labels) in enumerate(valid_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)

            f1  = f1_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
            valid_f1  += f1 * inputs.size(0)

            try:
                ra  = roc_auc_score(labels.data.view_as(predictions).cpu(), predictions.cpu())
                valid_ra  += ra * inputs.size(0)
            except ValueError:
                print("ROC-AUC ValueError")
                
                print(f"Validation Batch number: {i:03d}, Validation: Loss: {valid_loss/(j+1)/inputs.size(0):.4f}, Accuracy: {valid_acc/(j+1)/inputs.size(0):.4f}, F1: {valid_f1/(j+1)/inputs.size(0):.4f}, ROC-AUC: {valid_ra/(j+1)/inputs.size(0):.4f}")

    avg_train_loss = train_loss/train_data_size 
    avg_train_acc  = train_acc/float(train_data_size)
    avg_train_f1   = train_f1/float(train_data_size)
    avg_train_ra   = train_ra/float(train_data_size)

    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc  = valid_acc/float(valid_data_size)
    avg_valid_f1   = valid_f1/float(valid_data_size)
    avg_valid_ra   = valid_ra/float(valid_data_size)

    name = f"CarNet_{epoch:03d}"
    torch.save(model, f"Checkpoints/{name}.pth")
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    epoch_end = time.time()
    print(f"Epoch : {epoch:03d}, Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%, F1: {avg_train_f1:.4f}, ROC-AUC: {avg_train_ra:.4f}\n Validation : Loss : {avg_valid_loss:.4f}, Accuracy: {avg_valid_acc*100:.4f}%, F1: {avg_valid_f1:.4f}, ROC-AUC: {avg_valid_ra:.4f}, Time: {epoch_end-epoch_start:.4f}s")