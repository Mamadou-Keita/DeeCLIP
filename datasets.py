import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import random
import io
from transformers import  CLIPImageProcessor


def compress_image(image: Image.Image, quality: int = 85) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    compressed_image = Image.open(buffer)
    return compressed_image

def downsample_image(img, output_size=(112,112)):

    transform = transforms.Compose([
        transforms.Resize(output_size, interpolation=Image.BICUBIC)
    ])
    downsampled_image = transform(img)

    return downsampled_image

def apply_gaussian_blur(img, sigma=3):

    image_array = np.array(img)
    
    blurred_array = image_array.copy()
    blurred_array = Image.fromarray(blurred_array)
    blurred_array = blurred_array.filter(ImageFilter.GaussianBlur(sigma))
    
    return blurred_array


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms=None, perturbation=None):

        self.data = pd.read_csv(input_filename)
        self.transforms = transforms
        self.perturb = perturbation
        self.processor = CLIPImageProcessor(do_resize=False).from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(self.data.iloc[idx, 0])
        images = Image.open(self.data.iloc[idx, 0]).convert('RGB')
        label = torch.tensor(int(self.data.iloc[idx, 2])) 
        

        if self.perturb:
            pass

        if self.transforms:
            images = self.transforms(images)

        images = self.processor(images, return_tensors="pt")["pixel_values"][0]
        
        return images, label

class TripletDataset(Dataset):
    def __init__(self, real_csv, fake_csv, use_transforms, random_transforms):
        # Load data from CSV files
        self.real_data = pd.read_csv(real_csv)
        self.fake_data = pd.read_csv(fake_csv)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.random_transforms = random_transforms

        self.use_transforms = use_transforms

    def set_transforms(self, use_transforms):
        """Enable or disable transformations."""
        self.use_transforms = use_transforms

    def __len__(self):
        return len(self.real_data) + len(self.fake_data)

    def __getitem__(self, idx):
        labels = []
        # Randomly choose the anchor source
        if random.choice([True, False]):  
            # Anchor is a real sample
            anchor = self.real_data.iloc[idx % len(self.real_data), 0]
            labels.append(self.real_data.iloc[idx % len(self.real_data), 2])

            positive_idx = random.randint(0, len(self.real_data) - 1)
            positive = self.real_data.iloc[positive_idx, 0] 
            labels.append(self.real_data.iloc[positive_idx, 2])

            negative_idx = random.randint(0, len(self.fake_data) - 1)
            negative = self.fake_data.iloc[negative_idx, 0] 
            labels.append(self.fake_data.iloc[negative_idx, 2])
        else:
            # Anchor is a fake sample
            anchor = self.fake_data.iloc[idx % len(self.fake_data), 0]
            labels.append(self.fake_data.iloc[idx % len(self.fake_data), 2])  
            
            positive_idx = random.randint(0, len(self.fake_data) - 1)
            positive = self.fake_data.iloc[positive_idx, 0] 
            labels.append(self.fake_data.iloc[positive_idx, 2])
            
            negative_idx = random.randint(0, len(self.real_data) - 1)
            negative = self.real_data.iloc[negative_idx, 0] 
            labels.append(self.real_data.iloc[negative_idx, 2])

        # Prepare data for CLIP
        anchor_image = Image.open(anchor)
        positive_image = Image.open(positive)
        negative_image = Image.open(negative)

        if self.use_transforms:
            anchor_image = self.random_transforms(anchor_image)
            positive_image = self.random_transforms(positive_image)
            negative_image = self.random_transforms(negative_image)

        anchor_image = self.processor(images=anchor_image, return_tensors="pt")["pixel_values"][0]
        positive_image = self.processor(images=positive_image, return_tensors="pt")["pixel_values"][0]
        negative_image = self.processor(images=negative_image, return_tensors="pt")["pixel_values"][0]


        #label = torch.tensor([0.0]) if '0_real' in anchor else torch.tensor([1.0])
        
        return anchor_image, positive_image, negative_image, torch.tensor(labels)
