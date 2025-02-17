import argparse
import torch
import torch
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datasets import CsvDataset, RandomTransforms
import torchvision.transforms as transforms
from model import DeeCLIP
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatterMathtext

def count_parameters(model):
    """
    Counts the total number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model whose parameters are to be counted.
        
    Returns:
        int: Total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


#### options
parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str, help="Path to options test file.")
args = parser.parse_args()

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set random seed for NumPy
np.random.seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                            # transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                        ])


model = DeeCLIP(layer_indices=[1, 3, 5, 8, 10, 13, 15, 17, 19, 21, 22, 23]).to(device) #[1, 2, 3, 5, 7, 8, 10, 13, 15, 17, 19, 23]
#print(model)

model.load_state_dict(torch.load("weights/deeclip_weight_complete_with_lora_5.pth", map_location=device), strict=False)


model.eval() 
total_params = count_parameters(model)
print(f"Total parameters in the model: {total_params}")

all_accuracy = []

for file_name in ['progan.csv', 'cyclegan.csv', 'biggan.csv', 'stylegan.csv', 'gaugan.csv', 'stargan.csv', 
              'deepfake.csv', 'seeingdark.csv', 'san.csv', 'crn.csv', 'imle.csv', 'guided.csv', 
              'ldm_200.csv', 'ldm_200_cfg.csv', 'ldm_100.csv', 'glide_100_27.csv', 'glide_50_27.csv', 
              'glide_100_10.csv', 'dalle.csv']:

    datasetTest = CsvDataset('UniversalFakeDetect/test/'+file_name, None)
    test_dataloader = DataLoader(datasetTest, batch_size=8, shuffle=False)
            
    with torch.no_grad():  
        logits_genuine = []
        logits_forged = []
        for imgs, labels in tqdm(test_dataloader):

            imgs = imgs.to(device)   
            labels = labels.to(device)  
                    
            outputs = model(imgs, train=False)  
            probs = torch.sigmoid(outputs) 
            logits = outputs.squeeze().cpu().numpy()
             

            for i, label in enumerate(labels):
                if label == 0: 
                    logits_genuine.append(logits[i])  
                elif label == 1: 
                    logits_forged.append(logits[i])

    logits_genuine, logits_forged = np.array(logits_genuine), np.array(logits_forged)  
    fig, ax = plt.subplots()

    # Plot histograms with transparency
    ax.hist(logits_genuine, bins=150, alpha=0.5, label=file_name.split('.')[0] + ' real')
    ax.hist(logits_forged, bins=150, alpha=0.5, label=file_name.split('.')[0] + ' fake')

    # Set custom x-ticks
    #ax.set_xticks([-4, -2, 0, 2, 4])

    # Add legend in the top-right corner
    ax.legend(loc='upper right')

    # Save the plot
    plt.savefig('DeeCLIP'+file_name.split('.')[0] + '.png', dpi=300)
    
