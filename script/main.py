import torch
import cv2

import albumentations as A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.losses import DiceLoss

CSV_FILE = '../src/train.csv'
DATA_DIR = '../src/'


DEVICE = 'cuda'

EPOCHS = 25
LR = 0.003
IMAGE_SIZE = 320
BATCH_SIZE = 16

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

df = pd.read_csv(CSV_FILE)

train_images, test_images = train_test_split(df, test_size=0.2, random_state=42)

def get_train_augs():
   return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5)
   ])

def get_valid_augs():
   return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE),
   ])

def train_fn(data_loader, model, optimizer):
   model.train()
   total_loss = 0.0

   for images, masks in tqdm(data_loader):
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      optimizer.zero_grad()
      logits, loss = model(images, masks)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

   return total_loss / len(data_loader)

def valid_fn(data_loader, model):
   model.eval()
   total_loss = 0.0

   with torch.no_grad():
      for images, masks in tqdm(data_loader):
         images = images.to(DEVICE)
         masks = masks.to(DEVICE)

         logits, loss = model(images, masks)

         total_loss += loss.item()

   return total_loss / len(data_loader)      

class SegmentationDataset(Dataset):
   def __init__(self, df, augmentations):
      self.df = df
      self.augmentations = augmentations
   
   def __len__(self):
      return len(self.df)

   def __getitem__(self, idx):
      row = self.df.iloc[idx]
      image_path = row.images
      mask_path = row.masks

      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
      mask = np.expand_dims(mask, axis=-1)

      if self.augmentations:
         data = self.augmentations(image=image, mask=mask)
         image = data['image']
         mask = data['mask']
         image = np.transpose(image, (2,0,1)).astype(np.float32)
         mask = np.transpose(mask, (2,0,1)).astype(np.float32)

         image = torch.Tensor(image) / 255.0
         mask = torch.round(torch.Tensor(mask) / 255.0)
         
         return image, mask

class SegmentationModel(nn.Module):
   def __init__(self):
      super(SegmentationModel, self).__init__()
      self.arc = smp.Unet(encoder_name=ENCODER, enconder_weights=WEIGHTS, in_channels=3, classes=1, activation=None)

   del forward(self, images, masks=None):
      logits = self.arc(images)

   if masks != None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithlogitsLoss()(logits, masks)
      return logits, loss1 + loss2

   return logits

if __name__ == '__main__':

   trainset = SegmentationDataset(train_images, get_train_augs())
   testset = SegmentationDataset(test_images, get_valid_augs())

   trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
   testloader = DataLoader(testset, batch_size=BATCH_SIZE)

   model = SegmentationModel()
   model.to(DEVICE)

   optimizer = torch.optim.Adam(model.parameters(), lr=LR)

   best_valid_loss = np.Inf

   for i in range(EPOCHS):
      train_loss = train_fn(trainloader, model, optimizer)
      valid_loss = valid_fn(testloader, model)

      if valid_loss < best_valid_loss:
         torch.save(model.state_dict(), 'best_model.pt')
         print('MODEL-SAVED')
         best_valid_loss = valid_loss

      print(f'Epochs: {i+1} Train_loss: {train_loss} Valid_loss: {valid_loss}')








