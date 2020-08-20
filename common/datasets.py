import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom
from torch.utils.data.dataloader import DataLoader
import os

class NumpyDataset(Dataset):
    def __init__(self, X, y=None, xpath="./", ypath="./", transform=None, zoom=1, square=False):
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None
        self.xpath = xpath
        self.ypath = ypath
        self.transform = transform
        self.zoom = zoom
        self.square = square

    # Get file using filename and return float npy
    def _load(self, i, data, x=True):
        if x:
            path = self.xpath + "/" + data[i]
        else:
            path = self.ypath + "/" + data[i]
        
        arr = np.load(path).astype(np.single)
        arr = torch.from_numpy(arr).float()

        if self.transform:
            arr = self.transform(arr)

        return arr

    def __getitem__(self, i):
        x = self._load(i, self.X)
        x = zoom(x, self.zoom)
        
        if self.square:
            dim = x.shape[1]
            x = x[:dim]

        if self.y is not None:
            y = self._load(i, self.y, x=False)
            y = zoom(y, self.zoom)
                
            if self.square:
                y = y[:dim]
                
            return (x, y)
            
        return (x, None)

    def __len__(self): return len(self.X)

class NumpyClassDataset(Dataset):
    def __init__(self, class1, class2, class1path="./", class2path="./", transform=None, zoom1=1, zoom2=1, square=False):
        self.class1 = class1
        self.class2 = class2
        self.class1path = class1path
        self.class2path = class2path
        self.transform = transform
        self.num1 = len(class1)
        self.num2 = len(class2)
        self.zoom1 = zoom1
        self.zoom2 = zoom2
        self.square = square

    # Get file using filename and return float npy
    def _load(self, i):
        
        # PLEASE REFECTOR THIS LATER ITS GROSS
        if i < self.num1:
            path = self.class1path + "/" + self.class1[i]
            arr = np.load(path, allow_pickle=True).astype(np.single)
            arr = torch.from_numpy(arr).float()
            
            if self.transform:
                arr = self.transform(arr)
            
            arr = zoom(arr, self.zoom1)
            
        else:
            path = self.class2path + "/" + self.class2[i - self.num1]
            arr = np.load(path, allow_pickle=True).astype(np.single)
            arr = torch.from_numpy(arr).float()
            
            if self.transform:
                arr = self.transform(arr)
            
            arr = zoom(arr, self.zoom2)
            
        return arr

    def __getitem__(self, i):
        x = self._load(i)
        
        if self.square:
            dim = x.shape[1]
            x = x[:dim]
        
        if i < self.num1:
            y = torch.tensor([0, 1])
        else:
            y = torch.tensor([1, 0])
        return (x, y)

    def __len__(self): return self.num1 + self.num2

def get_generations(model, savepath, loadpath, transform=[], zoom=1, num=-1, bs=5, device="cpu"):
    xnames = os.listdir(loadpath)
    dset = NumpyDataset(xnames, xnames, xpath=loadpath, ypath=loadpath, transform=transform, zoom=zoom, square=True)
    loader = DataLoader(dset, batch_size=bs, shuffle=False)
    sample = np.load(loadpath + "/" + xnames[0])
    mean = sample.mean()
    std = sample.std()
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        i = 0
        for xs, _ in iter(loader):
            xs = torch.unsqueeze(xs, 1).to(device)
            
            outputs = model(xs)
            for output in outputs:
                output.mul_(std).add_(mean)
                output = output.cpu().detach().numpy()
                np.save(savepath + "/" + str(i) + ".npy", output[0])
                
                i += 1
                if i == num and i != -1:
                    break
                    
            if i == num and i != -1:
                break