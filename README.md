# DeeCLIP: A Robust Transformer-Based Approach for AI-Generated Image Detection

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

### TODO
We are working hard on the following items:

- [ ] Release ArXiv paper  
- [ ] Release training scripts  
- [x] Release inference scripts  
- [x] Release checkpoints

## Introduction

In this paper, we proposed DeeCLIP, a novel, robust, and generalizable transformer-based method for efficiently detecting AI-generated images. DeeCLIP incorporates two key components: (1) fine-tuning the CLIP-ViT image encoder using LoRA, a parameter-efficient fine-tuning technique, and (2) integrating both deep and shallow features from CLIP-ViT to preserve semantic alignment from deep layers while enriching representations with fine-grained details. In addition, we trained DeeCLIP end-to-end using triplet loss, which helps to better discriminate between authentic and AI-generated images by learning a more effective embedding space.

<p align="center">
  <img src="assets/comparison3.png" alt="Approach Image">
</p>

## Requirements
``` python
pip install -r requirements.txt
```

## Download Model Weights

To use the model, download the weight and save it in the `weights` folder.

### **Automatic Download (Command Line)**
Run the following command in your terminal:

```sh
mkdir -p weights && wget -O weights/deeclip_weight_complete_with_lora_5.pth "https://www.dropbox.com/scl/fi/ttiqnbxu8atz4on5gqvgd/deeclip_weight_complete_with_lora_5.pth?rlkey=6xznuvriabkqfdcofhi1pbihu&st=fk02k7hf&dl=1"
```

## SOTA Detection Methods

- [C2P-CLIP](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection)
- [RINE](https://github.com/mever-team/rine)
- [FatFormer](https://github.com/Michel-liu/FatFormer)
- [AntifakePrompt](https://github.com/nctu-eva-lab/antifakeprompt)
- [Bi-LORA](https://github.com/Mamadou-Keita/VLM-DETECT/)

## Evaluation
To run the test on specific dataset, use the following command:
```python
python testing.py
```

## T-SNE Plot of feature distribution

| ![](assets/tsne_plot_progan.png) | ![](assets/tsne_plot_stylegan.png) | ![](assets/tsne_plot_stargan.png) | ![](assets/tsne_plot_crn.png) |
|------------------------|------------------------|------------------------|------------------------|
| ![](assets/tsne_plot_imle.png) | ![](assets/tsne_plot_guided.png) | ![](assets/tsne_plot_glide_100_10.png) | ![](assets/tsne_plot_ldm_100.png) |


## Generalization Performance of Methods Trained on ProGAN

Results show accuracy (%) on real (🟢) and synthetic (🔴) data subsets.

| Methods   | Training Set | #params | 🟢 MS COCO | 🟢 Flickr | 🔴 ControlNet | 🔴 Dall3 | 🔴 DiffusionDB | 🔴 IF | 🔴 LaMA | 🔴 LTE | 🔴 SD2Inpaint | 🔴 SDXL | 🔴 SGXL | 🔴 SD3 | mAcc |
|-----------|-------------|---------|------------|-----------|--------------|----------|---------------|------|------|------|-------------|------|------|------|-----------|
| **FatFormer** | ProGAN | 493M | 33.97 | 34.04 | 28.27 | 32.07 | 28.10 | 27.95 | 28.67 | 12.37 | 22.63 | 31.97 | 22.23 | 35.91 | 28.18 |
| **RINE**      | ProGAN | 434M | 99.80 | 99.90 | 91.60 | 75.00 | 73.00 | 77.40 | 30.90 | 98.20 | 71.90 | 22.90 | 98.50 | 08.30 | 70.56 |
| **C2P-CLIP**  | ProGAN | 304M | 99.67 | 99.73 | 15.10 | 75.57 | 27.87 | 89.56 | 65.43 | 00.20 | 27.90 | 82.90 | 07.17 | 70.46 | 55.13 |
| **DecCLIP**   | ProGAN | 306M | 97.83 | 98.50 | 86.03 | 69.33 | 71.10 | 61.37 | 63.07 | 99.97 | 80.57 | 62.60 | 98.90 | 58.61 | **78.99** |


## :book: Citation
if you make use of our work, please cite our paper
```
@article{,
  title={DeeCLIP: A Robust Transformer-Based Approach for AI-Generated Image Detection},
  author={Keita, Mamadou and Hamidouche, Wassim and Bougueffa, Hassen and Hadid, Abdenour and Taleb-Ahmed, Abdelmalik},
  journal={},
  year={2025}
}
```
