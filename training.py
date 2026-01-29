
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import TripletDataset, RandomTransforms, CsvDataset
from model import DeeCLIP
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from losses import TripletLoss


def test_model(model, data_loader, device):
    model.eval() 

    all_test_labels = []
    all_test_preds = []
        
    with torch.no_grad():  

        for imgs, labels in tqdm(data_loader):

            imgs = imgs.to(device)  
            labels = labels.float().to(device)  
                
            outputs = model(imgs, train=False)  
            probs = torch.sigmoid(outputs) 

            predictions = (probs >= 0.5).float()

            all_test_preds.extend(predictions.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
                            
    accuracy = accuracy_score(all_test_labels, all_test_preds) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")



def check_and_print_layer_status(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} -> Trainable")
        else:
            print(f"Layer: {name} -> Frozen")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for NumPy
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeeCLIP(layer_indices=[1, 3, 5, 8, 10, 13, 15, 17, 19, 21, 22, 23]).to(device)
# model.load_state_dict(torch.load("deeclip_weight_complete_5.pth", map_location=device))

check_and_print_layer_status(model)



triplet_dataset = TripletDataset('./reals.csv','./fakes.csv', False, None)
data_loader = DataLoader(triplet_dataset, batch_size=8, shuffle=True)

datasetTest = CsvDataset("AntifakePromptDataset/DALLE2.csv", None)
test_dataloader = DataLoader(datasetTest, batch_size=8, shuffle=False)

# Loss functions
triplet_loss_fn = TripletLoss()
classification_loss_fn = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


# Training loop
num_epochs = 5



for epoch in range(num_epochs):

    model.train()
    total_loss = 0.0

    test_model(model, test_dataloader, device)

    for anchor_image, positive_image, negative_image,labels in tqdm(data_loader):

        # Move images and labels to device
        anchor_image, positive_image, negative_image = anchor_image.to(device), positive_image.to(device), negative_image.to(device)
        #images = images.squeeze(0).to(device)
        labels = torch.as_tensor(labels).flatten().to(device)


        optimizer.zero_grad()

        anchor_features, anchor_logits  = model(anchor_image)
        positive_features, positive_logits  = model(positive_image)
        negative_features, negative_logits  = model(negative_image)

        triplet_loss = triplet_loss_fn(anchor_features, positive_features ,negative_features)

        logits = torch.cat([anchor_logits, positive_logits, negative_logits], dim=0).view(-1)


        classification_loss = classification_loss_fn(logits, labels.float())

        total_batch_loss = triplet_loss + 2.0* classification_loss
        
        total_batch_loss.backward()


        optimizer.step()

        total_loss += total_batch_loss.item()

    # Print average loss per epoch
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    indice = str(epoch)
    torch.save(model.state_dict(), "extension/deeclip_weight_complete_"+indice+".pth")

test_model(model, test_dataloader, device)

torch.save(model.state_dict(), "extension/deeclip_weight_complete.pth")
torch.save(model, "extension/deeclip_model_complete_with_lora.pth")