import os
import torch
import cv2
from torch.utils.data import Dataset
import numpy as np

class UAVDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. LOAD IMAGE: Read the file from your folder
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # YOLO expects RGB
        
        # 2. RESIZE: Math to make it 640x640
        image = cv2.resize(image, (640, 640))
        
        # 3. NORMALIZE: Convert 0-255 pixels to 0.0-1.0 range
        # Math: x' = x / 255
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 4. LOAD LABELS: Read the .txt file for this image
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt'))
        
        # Handling background images (no drone)
        if not os.path.exists(label_path):
            targets = torch.zeros((0, 5)) # Empty target for background
        else:
            targets = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        return image, targets