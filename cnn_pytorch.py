import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob

# Dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            raise ValueError(f"Failed to read image: {self.image_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# CNN Model
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_data(data_path):
    image_paths = []
    labels = []
    for label in os.listdir(data_path):
        label_dir = os.path.join(data_path, label)
        if not os.path.isdir(label_dir):
            continue
        # Recursively find all image files (jpg, png, jpeg) in subdirectories
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            for img_path in glob.glob(os.path.join(label_dir, '**', ext), recursive=True):
                image_paths.append(img_path)
                labels.append(label)
    return image_paths, labels

def filter_valid_images(image_paths, labels):
    valid_image_paths = []
    valid_labels = []
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        if i % 100 == 0:
            print(f"Checking image {i}/{len(image_paths)}: {img_path}")
        img = cv2.imread(img_path)
        if img is not None:
            valid_image_paths.append(img_path)
            valid_labels.append(label)
    return valid_image_paths, valid_labels

def main():
    data_path = r"E:\project\new\Two-Way-Sign-Language-Translator\data_1kag"
    image_paths, labels = load_data(data_path)
    image_paths, labels = filter_valid_images(image_paths, labels)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    pd.DataFrame({'Label': le.classes_, 'Encoded': range(num_classes)}).to_csv('label_encoded.csv', index=False)

    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels_encoded, test_size=0.1, random_state=42)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    train_dataset = SignLanguageDataset(X_train, y_train, transform=transform)
    val_dataset = SignLanguageDataset(X_val, y_val, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == lbls).sum().item()
            total += lbls.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")
        if train_acc >= 0.9:
            print(f"Early stopping: Training accuracy reached {train_acc:.4f} at epoch {epoch+1}")
            break
    torch.save(model.state_dict(), 'trained_model.pth')
    print('Model saved as trained_model.pth')

if __name__ == '__main__':
    main()