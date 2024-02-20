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
import tqdm as tqdm
import glob
import time
from models import *
from datasets import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print("device :", device)

# images
images_dir = "trainset"

# Load clinical data
data = pd.read_csv("clinical_annotation.csv")
data_train_val = data[data["LABEL"]!=-1]
data_test = data[data["LABEL"]==-1]

data_train, data_val = train_test_split(data_train_val, test_size=0.2)

clinical_train = ClinicalData(data_train)
clinical_val = ClinicalData(data_val)

image_train = ImageDataset(data_train, images_dir)
image_val = ImageDataset(data_val, images_dir)

dataloader_MLP = DataLoader(clinical_train, batch_size=512, shuffle=True)
dataloader_CNN = DataLoader(image_train, batch_size=512, shuffle=True)


mlp = MLP().to(device)
feature_extractor = ConvolutionalFeatureExtractor(in_channels=3, num_classes=1, K=8).to(device)

optimizerMLP = torch.optim.Adam(params = mlp.parameters(), lr=1e-1)
criterionMLP = nn.BCELoss()

optimizerCNN = torch.optim.Adam(params = feature_extractor.parameters(), lr=1e-3) #, betas=(0.9, 0), weight_decay=5e-4)
criterionCNN = nn.BCELoss()

## Training MLP ##
n_epochs = 1000
losses = []
for _ in tqdm.tqdm(range(n_epochs)):
    for i, (x, y) in enumerate(dataloader_MLP) :
        y_pred = mlp(x.to(device))
        optimizerMLP.zero_grad()
        loss = criterionMLP(y_pred, y.to(device))
        losses.append(loss.item())
        loss.backward()
        optimizerMLP.step()

## Saving the MLP
torch.save(mlp, "MLP.pth")

dataloader_val = DataLoader(clinical_val, batch_size=64, shuffle=True)
acc = 0
balanced_acc = 0 
predictions = []
true_labels = []
with torch.no_grad():
    for i, (x, y) in tqdm.tqdm(enumerate(dataloader_val)) :
        y_pred = mlp(x.to(device))
        
        predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
        true_labels += list(y.numpy())
print("MLP Validation")
print("Accuracy score : {:.2f}".format(accuracy_score(predictions, true_labels)))
print("Balanced accuracy score : {:.2f}".format(balanced_accuracy_score(predictions, true_labels)))


## Training CNN ##

n_epochs = 10 # 50
losses = []
for k in tqdm.tqdm(range(n_epochs)):
    total_loss = 0
    print("Start Epoch "+str(k))
    start = time.time()
    for i, (x, y) in tqdm.tqdm(enumerate(dataloader_CNN)) :
        y_pred = feature_extractor(x.to(device))
        optimizerCNN.zero_grad()
        loss = criterionCNN(y_pred, y.to(device))
        losses.append(loss.item())
        total_loss+=losses[-1]
        loss.backward()
        optimizerCNN.step()
        
    end = time.time()
    print("Epoch "+str(k)+" finished.")
    print("Loss on training set :", total_loss)
    ## Saving the CNN
    torch.save(feature_extractor, "CNN" + str(k) +".pth")

        
np.save("Training loss", losses)

# Validation
dataloader_val = DataLoader(image_val, batch_size=32, shuffle=True)

acc = 0
balanced_acc = 0 
predictions = []
true_labels = []
with torch.no_grad():
    for i, (x, y) in tqdm.tqdm(enumerate(dataloader_val)) :
        y_pred = feature_extractor(x.to(device))
        
        predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
        true_labels += list(y.numpy())
print("CNN Validation")
print("Accuracy score : ", accuracy_score(predictions, true_labels))
print("Balanced accuracy score : ", balanced_accuracy_score(predictions, true_labels))
        
## Prediction and softvoting ##