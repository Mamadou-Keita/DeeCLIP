import argparse
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
from datasets import CsvDataset
from model import DeeCLIP
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

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

model = DeeCLIP(layer_indices=[1, 3, 5, 8, 10, 13, 15, 17, 19, 21, 22, 23]).to(device) 

model.load_state_dict(torch.load("weights/deeclip_weight_complete_with_lora_5.pth", map_location=device), strict=False)
model.eval() 


# Define the list of file names (test sets)
file_names = ['progan.csv', 'cyclegan.csv', 'biggan.csv', 'stylegan.csv', 'gaugan.csv', 'stargan.csv', 
              'deepfake.csv', 'seeingdark.csv', 'san.csv', 'crn.csv', 'imle.csv', 'guided.csv', 
              'ldm_200.csv', 'ldm_200_cfg.csv', 'ldm_100.csv', 'glide_100_27.csv', 'glide_50_27.csv', 
              'glide_100_10.csv', 'dalle.csv']

for file_name in file_names:
    # Variables to store t-SNE data
    all_features = []
    all_labels = []

    datasetTest = CsvDataset('UniversalFakeDetect/test/' + file_name, None, None)
    test_dataloader = DataLoader(datasetTest, batch_size=1, shuffle=False)

    with torch.no_grad():  

        for imgs, labels in tqdm(test_dataloader):

            imgs = imgs.to(device)  
            labels = labels.to(device) 
                    
            _, fused_features,_ = model(imgs, train=False)

            # Reduce n_tokens dimension using mean pooling
            fused_features = fused_features.mean(dim=1).cpu().numpy()
            # flattened_features = fused_features.view(-1).cpu().numpy()
            all_features.append(fused_features)
            all_labels.append(labels.cpu().numpy())

    all_fused_features = np.vstack(all_features)  # Shape: (N_samples, 768)
    all_labels = np.array(all_labels).astype(int)

    # Run t-SNE on fused features
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(all_fused_features)

    # Fix boolean masks by flattening
    real_mask = (all_labels == 0).squeeze()  # Shape should be (N_samples,)
    fake_mask = (all_labels == 1).squeeze()  # Shape should be (N_samples,)

    # Debugging: Print the shapes
    print("Real Mask Shape:", real_mask.shape)  # Should be (N_samples,)
    print("Fake Mask Shape:", fake_mask.shape)  # Should be (N_samples,)
    print("t-SNE Output Shape:", X_2d.shape)  # Should be (N_samples, 2)

    # Apply boolean indexing correctly
    real_points = X_2d[real_mask]  # Corrected indexing
    fake_points = X_2d[fake_mask]  # Corrected indexing

    # Debugging: Check selected points
    print("Real Points Shape:", real_points.shape)  # Should be (num_real_samples, 2)
    print("Fake Points Shape:", fake_points.shape)  # Should be (num_fake_samples, 2)

    # Plot t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.title(file_name.split(".")[0])
    plt.scatter(real_points[:, 0], real_points[:, 1], label="Real", color="green", alpha=0.7)
    plt.scatter(fake_points[:, 0], fake_points[:, 1], label="Fake", color="red", alpha=0.7)

    plt.legend()
    plt.grid(False) # Remove grid from the plot

    plot_filename = f'tsne_plot_{file_name.split(".")[0]}.png'
    plt.savefig(plot_filename)
    # plt.show()