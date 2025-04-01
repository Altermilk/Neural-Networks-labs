import csv
import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import numpy as np


train_data_labels = {}
csv_path = 'Training_set.csv'
train_dir = 'train/'  

with open(csv_path, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  
    for row in reader:
        filename, label = row[0], row[1]
        if label not in train_data_labels:
            train_data_labels[label] = []
        train_data_labels[label].append(filename)

species = list(train_data_labels.keys())  # Все возможные классы


class CustomDataset(Dataset):
    def __init__(self, data_dict, root_dir, transform=None):
        self.data = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(data_dict.keys())}

        for label, filenames in data_dict.items():
            for filename in filenames:
                img_path = os.path.join(root_dir, filename)
                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_class_weights():
        label_to_idx = {label: idx for idx, label in enumerate(species)}
        labels = [label_to_idx[label] for label_list in train_data_labels.values() for label in label_list]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights

# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(30),
#     transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#     transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomDataset(train_data_labels, train_dir, transform=transform)
train_size = int(0.8 * len(train_dataset))  
test_size = len(train_dataset) - train_size  
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)