import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import glob

mean = [0.84, 0.72, 0.71]
std = [0.19, 0.21,  0.09]
class ClinicalData(Dataset):
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise Exception("data shoud be a pandas.DataFrame")
            
        self.label = data["LABEL"].values 
        self.counts = data["LYMPH_COUNT"].values
        self.ages = []
        for bod in data["DOB"]:
            self.ages.append(2024 - int(bod[-4:]))
        
        self.features = torch.Tensor(list(zip(self.counts, self.ages)))
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        y = self.label[idx]
        x = self.features[idx]
        
        return x, torch.Tensor([y])
    
class ImageDataset(Dataset):
    def __init__(self, data, root_dir):
        if not isinstance(data, pd.DataFrame):
            raise Exception("data shoud be a pandas.DataFrame")
        if not isinstance(root_dir, str):
            raise Exception("root_dir shoud be a string")
            
        self.id = data["ID"].values        
        self.label = data["LABEL"].values 
        self.dir = root_dir

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        ID, y = self.id[idx], self.label[idx]
        files = glob.glob(self.dir + "/" + ID + "/*")
        b = len(files)
        X = torch.zeros((b, 3, 224, 224))
        for i in range(b):
            image = Image.open(files[i])
            image = np.array(image)
            image = image / 255 
            X[i,:,:,:] = torch.Tensor(image.transpose(-1, 0, 1))
        return X, torch.Tensor([y])

transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class PatientDataset(Dataset):
    def __init__(self, data, root_dir):
        if not isinstance(data, pd.DataFrame):
            raise Exception("data shoud be a pandas.DataFrame")
        if not isinstance(root_dir, str):
            raise Exception("root_dir shoud be a string")
            
        self.id = data["ID"].values        
        self.label = data["LABEL"].values 
        self.dir = root_dir

        self.counts = data["LYMPH_COUNT"].values
        self.ages = []
        for bod in data["DOB"]:
            self.ages.append(2024 - int(bod[-4:]))
        
        self.features = torch.Tensor(list(zip(self.counts, self.ages)))

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        ID, y = self.id[idx], self.label[idx]
        x = self.features[idx]
        files = glob.glob(self.dir + "/" + ID + "/*")
        b = len(files)
        X = torch.zeros((b, 3, 224, 224))
        for i in range(b):
            image = Image.open(files[i])
            image = np.array(image)
            image = image / 255 
            X[i,:,:,:] = torch.Tensor(image.transpose(-1, 0, 1))
        return (X, x), torch.Tensor([y])
