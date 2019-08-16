import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
import numpy as np
from PIL import Image
import sys
import numpy as np
import math
import json
import random


class Person_val(Dataset):
    def __init__(self, data_dir="./val", mode="train"):
        
        super(Person_val, self).__init__()
        
        self.data_dir = data_dir + "/"
        
        movie_names = sorted([file for file in os.listdir(data_dir) if file.startswith('tt')])
        
        self.candidate_datas = []
        self.cast_datas = []
        self.len = len(movie_names)
        
        
        self.move_dirs = []
        self.candidate_names = []
        self.cast_names = []
        
        # load all json file.
        for i in range(len(movie_names)):
            
            movie_dir = self.data_dir + movie_names[i]
            
            self.move_dirs.append(movie_dir)
            candidate_dir_name = movie_dir + "/candidates/"
            cast_dir_name = movie_dir + "/cast/"
        
        
            candidate_name = sorted([file for file in os.listdir(candidate_dir_name)])
            cast_name = sorted([file for file in os.listdir(cast_dir_name)])
            
            
            for j in range(len(candidate_name)):
                candidate_name[j] = candidate_dir_name+ candidate_name[j]
            for j in range(len(cast_name)):
                cast_name[j] = cast_dir_name+ cast_name[j]
                
            self.candidate_names.append(candidate_name)
            self.cast_names.append(cast_name)
        
    def __getitem__(self, index):
        
        cast_name = self.cast_names[index]
        candidate_name = self.candidate_names[index]

        return cast_name, candidate_name
    
    def __len__(self):
        return  self.len

class Image_val(Dataset):
    def __init__(self, filenames):
        self.filenames = np.array(filenames)
        
        self.len = len(filenames)
        self.transform = transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.ToTensor(), 
              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
         ])
        
    def __getitem__(self, index):
        
        fname = self.filenames[index][0]
        
        #print(fname)
        img = Image.open('./' + fname)
        
#         (w, h)= img.size
 
#         box = (0, 0, w, w)
#         img = img.crop(box)
        img = self.transform(img)
        
        return img
    
    def __len__(self):
        return  self.len
        
'''
data = Person_val(data_dir="./val")
print(len(data))

data = iter(data)
print( next(data) )
error
'''

