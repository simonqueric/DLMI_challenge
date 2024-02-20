import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import glob

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
            
        self.id_label = {}
        for Id, label in zip(data["ID"], data["LABEL"]):
            self.id_label[Id] = label
        
        self.images = []
        for Id in data["ID"] :
            self.images += glob.glob(root_dir+"/"+Id+"/*")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_name = self.images[idx]
        sp = file_name.split("/")
        Id = sp[1] 
        image, y = plt.imread(file_name), self.id_label[Id]
        return torch.Tensor(image.transpose(-1, 0, 1)), torch.Tensor([y])