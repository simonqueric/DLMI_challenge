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

data_train, data_val = train_test_split(data_train_val, test_size=0.33)
print("Size training set :", len(data_train))
print("Size validation set :", len(data_val))
print("Size test set :", len(data_test))
print("Size test set :", len(glob.glob("testset/*")) - 1)

clinical_train = ClinicalData(data_train)
clinical_val = ClinicalData(data_val)

# image_train = ImageDataset(data_train, images_dir)
# image_val = ImageDataset(data_val, images_dir)

image_train = PatientDataset(data_train, images_dir)
image_val = PatientDataset(data_val, images_dir)

print("Number of training images :", len(image_train))
print("Number of validation images :", len(image_val))

train_loader_MLP = DataLoader(clinical_train, batch_size=512, shuffle=True)
val_loader_MLP = DataLoader(clinical_val, batch_size=512, shuffle=True)
train_loader_CNN = DataLoader(image_train, batch_size=1, shuffle=True)
val_loader_CNN = DataLoader(image_val, batch_size=1, shuffle=True)
## Rename train_loader_MLP, val_loader_MLP
## Validation on val_loader_MLP
## add model.eval(), model.train()

mlp = MLP().to(device)
feature_extractor = ConvolutionalFeatureExtractor(in_channels=3, num_classes=1, K=8).to(device)

#feature_extractor.load_state_dict(torch.load("CNN49.pth"))

optimizerMLP = torch.optim.Adam(params = mlp.parameters(), lr=1e-1)
criterionMLP = nn.BCELoss()

optimizerCNN = torch.optim.Adam(params = feature_extractor.parameters(), lr=1e-3) #, betas=(0.9, 0), weight_decay=5e-4)
criterionCNN = nn.BCELoss()

## Training MLP ##
n_epochs = 1000
losses = []
best_loss = np.inf
for _ in tqdm.tqdm(range(n_epochs)):
    mlp.train()
    for i, (x, y) in enumerate(train_loader_MLP) :
        y_pred = mlp(x.to(device))
        optimizerMLP.zero_grad()
        loss = criterionMLP(y_pred, y.to(device))
        losses.append(loss.item())
        loss.backward()
        optimizerMLP.step()
    
    mlp.eval()
    total_loss = 0
    with torch.no_grad() :
        for i, (x, y) in enumerate(val_loader_MLP) :
            y_pred = mlp(x.to(device))
            loss = criterionMLP(y_pred, y.to(device))
            total_loss+=loss.item()
        total_loss /= i+1
    if total_loss < best_loss : 
        torch.save(mlp.state_dict(), "checkpoints/MLP.pth")
        best_loss = total_loss

mlp = MLP().to(device)
mlp.load_state_dict(torch.load("checkpoints/MLP.pth"))
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
print("Accuracy score : {:.2f}".format(accuracy_score(true_labels, predictions)))
print("Balanced accuracy score : {:.2f}".format(balanced_accuracy_score(true_labels, predictions)))


## Training CNN ##

n_epochs = 50
losses = []
best_loss = np.inf
best_accuracy = 0
for k in tqdm.tqdm(range(n_epochs)):
    feature_extractor.train()
    total_loss = 0
    print("Start Epoch "+str(k))
    for i, (x, y) in tqdm.tqdm(enumerate(train_loader_CNN)) :
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = feature_extractor(x.to(device))
        optimizerCNN.zero_grad()
        loss = criterionCNN(y_pred, y.to(device))
        losses.append(loss.item())
        total_loss+=losses[-1]
        loss.backward()
        optimizerCNN.step()
        
    print("Epoch "+str(k)+" finished. \n")
    print("Loss on Training Set : {:.3f}".format(total_loss/(i+1)))
    
    print("Validation")
    total_loss = 0
    feature_extractor.eval()
    # with torch.no_grad():
    #     for i, (x, y) in tqdm.tqdm(enumerate(val_loader_CNN)) :
    #         y_pred = feature_extractor(x.to(device))
    #         loss = criterionCNN(y_pred, y.to(device))
    #         total_loss+=losses[-1]
    # total_loss /= i+1 
    # print("Loss on Validation set : {:.3f}".format(total_loss))
    # if total_loss < best_loss :
    #     torch.save(feature_extractor.state_dict(), "checkpoints/CNN_best_model.pth")
    #     best_loss = total_loss
    balanced_acc = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for i, (x, y) in tqdm.tqdm(enumerate(val_loader_CNN)) :
            x = x.squeeze(0)
            y = y.squeeze(0)
            y_pred = feature_extractor(x.to(device))
            predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
            true_labels += list(y.cpu().numpy())
    print(predictions, true_labels)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)      
    print("CNN Validation")
    print("Balanced accuracy score on validation test: {:.2f}".format(balanced_acc))
    if balanced_acc > best_accuracy:
        torch.save(feature_extractor.state_dict(), "checkpoints/CNN_best_model.pth")
        best_accuracy = balanced_acc

        
np.save("training_loss", losses)

# Validation
# dataloader_val = DataLoader(image_val, batch_size=32, shuffle=True)

# acc = 0
# balanced_acc = 0 
# predictions = []
# true_labels = []
# with torch.no_grad():
#     for i, (x, y) in tqdm.tqdm(enumerate(dataloader_val)) :
#         y_pred = feature_extractor(x.to(device))
        
#         predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
#         true_labels += list(y.cpu().numpy())
# print("CNN Validation")
# print("Accuracy score : ", accuracy_score(predictions, true_labels))
# print("Balanced accuracy score : ", balanced_accuracy_score(predictions, true_labels))
        
## Softvoting : Validation ##

feature_extractor = ConvolutionalFeatureExtractor(in_channels=3, num_classes=1, K=8).to(device)
feature_extractor.load_state_dict(torch.load("checkpoints/CNN_best_model.pth"))

patients = []
predictions = []
true_labels = []
for patient in tqdm.tqdm(data_val["ID"].values) :
    patients.append(patient)
    label, dob, count = data_val[data_val["ID"]==patient]["LABEL"].values[0], data_val[data_val["ID"]==patient]["DOB"].values[0], data_val[data_val["ID"]==patient]["LYMPH_COUNT"].values[0]
    true_labels.append(label)
    
    age = 2024 - int(dob[-4:])
    clinical_features = torch.Tensor([age, count]).to(device)
    y_MLP = 1.*(mlp(clinical_features)>0.5)
    y_CNN = 0
    N = len(glob.glob(images_dir+"/"+patient+"/*"))
    for file_image in glob.glob(images_dir+"/"+patient+"/*") :
        image = plt.imread(file_image)
        image = torch.Tensor(image.transpose(-1, 0, 1)).to(device)
        y_pred = feature_extractor(image[None,:])
        y_CNN+=y_pred
        
    y_CNN /= N
    y_CNN = y_CNN[0, 0]
    y_pred = 1.*((y_CNN + y_MLP)/2 > 0.5)
    predictions.append(y_pred.cpu().numpy())

print("Soft Voting Validation")
print("Accuracy score : ", accuracy_score(true_labels, predictions))
print("Balanced accuracy score : ", balanced_accuracy_score(true_labels, predictions))
    
## Softvoting : Prediction ##
images_dir = "testset"
patients = []
predictions = []
for patient in (data_test["ID"].values) :
    patients.append(patient)
    _, dob, count = data_test[data_test["ID"]==patient]["LABEL"].values[0], data_test[data_test["ID"]==patient]["DOB"].values[0], data_test[data_test["ID"]==patient]["LYMPH_COUNT"].values[0]
    #true_labels.append(label)
    
    age = 2024 - int(dob[-4:])
    clinical_features = torch.Tensor([count, age]).to(device)
    y_MLP = mlp(clinical_features)
    y_CNN = 0
    N = len(glob.glob(images_dir+"/"+patient+"/*"))
    for file_image in glob.glob(images_dir+"/"+patient+"/*") :
        image = plt.imread(file_image)
        image = torch.Tensor(image.transpose(-1, 0, 1)).to(device)
        y_pred = feature_extractor(image[None,:]).to(device)
        y_CNN+=y_pred
    
    y_CNN /= N
    y_CNN = y_CNN[0, 0]
    y_pred = 1.*((y_CNN + y_MLP)/2>0.5)
    predictions.append(y_pred.cpu().numpy()[0])

submission = pd.DataFrame()
submission["Id"] = patients
submission["Predicted"] = predictions
submission.to_csv("submission.csv", sep=",", index=False)