import argparse
import torch
import os
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from tqdm import tqdm
from datasets import CsvDataset, RandomTransforms
import torchvision.transforms as transforms
from model import DeeCLIP
from torch.utils.data import DataLoader
import numpy as np

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_test = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

model = DeeCLIP(layer_indices=[1, 3, 5, 8, 10, 13, 15, 17, 19, 21, 22, 23]).to(device) #[1, 2, 3, 5, 7, 8, 10, 13, 15, 17, 19, 23]

model.load_state_dict(torch.load("weights/deeclip_weight_complete_with_lora_5.pth", map_location=device), strict= False)


model.eval() 
print(count_parameters(model))

all_accuracy = []
all_ap = []

folders = [
    # 'progan.csv','cyclegan.csv','biggan.csv','stylegan.csv','gaugan.csv','stargan.csv',
    # 'deepfake.csv','seeingdark.csv','san.csv','crn.csv','imle.csv','guided.csv','ldm_200.csv',
    # 'ldm_200_cfg.csv','ldm_100.csv', 'glide_100_27.csv','glide_50_27.csv','glide_100_10.csv','dalle.csv'
     'COCO', 'flickr30k_224', 'Control_COCO', 'dalle3', 'DiffusionDB',  'IF',
       'lama_224','lte_SR4_224' 'SD2Inpaint_224', 'SDXL', 'SGXL', 'SD3'
]

for file_name in folders:

        
    datasetTest = CsvDataset('./'+file_name+'.csv', None, None)
    test_dataloader = DataLoader(datasetTest, batch_size=32, shuffle=True)

    all_test_labels = []
    all_test_preds = []
                
    with torch.no_grad():  

        for imgs, labels in tqdm(test_dataloader):

            imgs = imgs.to(device)  
            labels = labels.float().to(device)  
                        
            _, _, outputs = model(imgs, train=False)
            probs = torch.sigmoid(outputs) 

            all_test_preds.extend(probs.float().cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())


    all_test_labels, all_test_preds = np.array(all_test_labels), np.array(all_test_preds)
                                    
    accuracy = accuracy_score(all_test_labels, all_test_preds > 0.5) * 100
    ap = average_precision_score(all_test_labels, all_test_preds) * 100

    all_accuracy.append(accuracy)
    all_ap.append(ap)


    print(f"Test Accuracy for {file_name}: {accuracy:.2f}% / Test AP: {ap:.2f}%")

    
print(f"Model : {np.mean(all_accuracy):.2f}% /  {np.mean(all_ap):.2f}%")
