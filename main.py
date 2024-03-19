import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from torchvision import models
import tqdm 
import glob
import time
from models import *
from datasets import *
from resnet import *
import argparse




## DEFAULT PARAMETERS ##

lr_CNN = 1e-3 # 1e-4, 5e-4, 1e-3
epochs_CNN = 10
lr_MLP = 1e-1
epochs_MLP = 1000
device = "cuda:1"

## Config ##

config = argparse.ArgumentParser(description="Configuration")
config.add_argument("--lr_CNN", type=float, default=lr_CNN)
config.add_argument("--lr_MLP", type=float, default=lr_MLP)
config.add_argument("--epochs_CNN", type=int, default=epochs_CNN)
config.add_argument("--epochs_MLP", type=int, default=epochs_MLP)
config.add_argument("--device", type=str, default=device)

parameters = config.parse_args()

print(parameters)

## DEVICE ##

device = torch.device(parameters.device if torch.cuda.is_available() else "cpu")

print("device :", device)

# images
images_dir = "trainset"

# Load clinical data
data = pd.read_csv("clinical_annotation.csv")
data_train_val = data[data["LABEL"]!=-1]
data_test = data[data["LABEL"]==-1]


# Analysis of the data
positive_label = data[data["LABEL"]==1]["ID"].values
negative_label = data[data["LABEL"]==0]["ID"].values

nb_positive = 0 
nb_negative = 0
for ID in positive_label :
    nb_positive += len(glob.glob("trainset/"+ID+"/*"))
for ID in negative_label :
    nb_negative += len(glob.glob("trainset/"+ID+"/*"))

print("negative samples :", nb_negative)
print("positive samples :", nb_positive)

data_train, data_val = train_test_split(data_train_val, test_size=0.3)  # 0.3 ? 0.4 ?, random_state=42) # test_size = 0.25
print("Size training set :", len(data_train))
print("Size validation set :", len(data_val))
print("Size test set :", len(data_test))
print("Size test set :", len(glob.glob("testset/*")) - 1)

# print(data_val)
clinical_train = ClinicalData(data_train)
clinical_val = ClinicalData(data_val)

#image_train = ImageDataset(data_train, images_dir)
#image_val = ImageDataset(data_val, images_dir)

image_train = PatientDataset(data_train, images_dir)
image_val = PatientDataset(data_val, images_dir)

(X, x), y = next(iter(image_train))
print("img shape :", X.shape)
print("max, min :", torch.max(X), torch.min(X))

# print("Number of training images :", len(image_train))
# print("Number of validation images :", len(image_val))

train_loader_MLP = DataLoader(clinical_train, batch_size=512, shuffle=True)
val_loader_MLP = DataLoader(clinical_val, batch_size=512, shuffle=True)

batch_size = 64
train_loader_CNN = DataLoader(image_train, batch_size=1, shuffle=True)
val_loader_CNN = DataLoader(image_val, batch_size=1, shuffle=True)


## Rename train_loader_MLP, val_loader_MLP
## Validation on val_loader_MLP
## add model.eval(), model.train()

mlp = MLP().to(device)
K = 16 # 4 8 16 32 64


resnet = models.resnet18(pretrained=False) #.to(device)
#resnet = models.resnet34(pretrained=False) #.to(device)
cnn = ModifiedResNet(resnet).to(device)

#feature_extractor.load_state_dict(torch.load("checkpoints/CNN0.0005_25.pth"))

optimizerMLP = torch.optim.Adam(params = mlp.parameters(), lr=parameters.lr_MLP)
criterionMLP = nn.BCELoss()


optimizerCNN = torch.optim.Adam(params = cnn.parameters(), lr=parameters.lr_CNN)#, betas=(0.9, 0), weight_decay=5e-4)
criterionCNN = nn.BCELoss()

# lmbda = lambda epoch: 0.99
# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizerCNN, lr_lambda=lmbda)
# Training MLP ##
n_epochs = parameters.epochs_MLP
losses = []
best_loss = np.inf


def train(epoch, print_every, name, model, optimizer, criterion, trainloader, valloader) :
    if epoch % print_every == 0 :
        print('\nEpoch: %d' % epoch)

    global best_acc
    model.train()
    for i, (x, y) in enumerate(trainloader) :
        
        if name=="CNN":
            image, clinical = x 
            clinical = clinical.to(device)
            image = image.squeeze(0).to(device)
            y = y.squeeze(0).to(device)
            optimizer.zero_grad()
            y_pred = model(image, clinical)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        else : 
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad() :
        for i, (x, y) in enumerate(valloader) :
            if name=="CNN":
                image, clinical = x 
                clinical = clinical.to(device)
                image = image.squeeze(0).to(device)
                y = y.squeeze(0).to(device)
                y_pred = model(image, clinical)
            else : 
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
            predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
            true_labels += list(y.cpu().numpy())
        balanced_acc = balanced_accuracy_score(true_labels, predictions)      
    
    if epoch % print_every == 0 :
        print('\nEpoch: %d' % epoch, 'finished')
        print("balanced accuracy : {:.2f}".format(balanced_acc))

    if balanced_acc > best_acc : 
        print("Accuracy improved")
        torch.save(model.state_dict(), "checkpoints/" + name + "_"+ "epoch_" + str(epoch) +".pth")
        best_acc = balanced_acc

best_acc = 0
print_every = 100
for epoch in tqdm.tqdm(range(n_epochs)):
   train(epoch, print_every, "MLP", mlp, optimizerMLP, criterionMLP, train_loader_MLP, val_loader_MLP)



# Training CNN ##

n_epochs = parameters.epochs_CNN
losses = []
best_loss = np.inf
best_accuracy = 0
print_every = 1
#loss = 0
# batch_size = 2
# count = 0
#Loss = 0

for epoch in tqdm.tqdm(range(n_epochs)):
   train(epoch, print_every, "CNN", cnn, optimizerCNN, criterionCNN, train_loader_CNN, val_loader_CNN)

for k in tqdm.tqdm(range(n_epochs)):
    cnn.train()
    total_loss = 0
    print("Start Epoch "+str(k))
    for i, (x, X, y) in tqdm.tqdm(enumerate(train_loader_CNN)) :
        # count+=1
        X = X.squeeze(0)
        y = y.squeeze(0)
        optimizerCNN.zero_grad()
        y_pred = cnn(X.to(device), x.to(device))
        loss = criterionCNN(y_pred, y.to(device)) #/batch_size
        #Loss+=loss
        
        losses.append(loss.item())
        total_loss+=losses[-1]  
        loss.backward()
        optimizerCNN.step()
        count = 0
        loss = 0
    print("Epoch "+str(k)+" finished. \n")
    print("Loss on Training Set : {:.3f}".format(total_loss/(i+1)))
    
    print("Validation")
    total_loss = 0
    cnn.eval()
    predictions = []
    true_labels = []
    logits = []
    with torch.no_grad():
        for i, (x, X, y) in tqdm.tqdm(enumerate(val_loader_CNN)) :
            X = X.squeeze(0)
            y = y.squeeze(0)
            y_pred = cnn(X.to(device), x.to(device))
            loss = criterionCNN(y_pred, y.to(device))
            total_loss+=loss.item()
            logits += list(y_pred.detach().cpu().numpy().astype(float))
            predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
            true_labels += list(y.cpu().numpy())

    total_loss /= i+1 
    print("Loss on Validation set : {:.3f}".format(total_loss))
    # if total_loss < best_loss :
    #    print("Loss decreased")
    #    torch.save(cnn.state_dict(), "checkpoints/CNN" +str(parameters.lr_CNN) +"_" + str(parameters.epochs_CNN) +".pth")
    #    best_loss = total_loss
    # print("true labels", true_labels)
    # print("predictions", predictions)
    # print("logits", logits)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)      
    print("CNN Validation")
    if balanced_acc > best_accuracy :
        print("Accuracy improved")
        torch.save(cnn.state_dict(), "checkpoints/CNN_"  + str(parameters.lr_CNN) + "_"+ str(parameters.epochs_CNN)+".pth")
        best_accuracy = balanced_acc
    print("Accuracy score on validation set : {:.2f}".format(accuracy_score(true_labels, predictions)))
    print("Balanced accuracy score on validation set: {:.2f}".format(balanced_acc))
    # scheduler.step()
        
np.save("training_loss", losses)
