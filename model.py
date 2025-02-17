import torch
from transformers import CLIPVisionModel
from PIL import Image
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import TripletLoss

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.scale = dim ** -0.5  # Scale factor for attention scores

    def forward(self, query, key, value):
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, value)
        return output

class MultiLayerPerceptron(nn.Module):

  def __init__(self, inFeatures, hiddenFeatures, outFeatures, dropout= 0.):
    super().__init__()
    
    self.fc1 = nn.Linear(inFeatures, hiddenFeatures)
    self.activation = nn.GELU()
    self.fc2 = nn.Linear(hiddenFeatures, outFeatures)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):

    x = self.fc1(x)
    x = self.activation(x)
    x = self.dropout(x)
      
    x = self.fc2(x)
    x = self.dropout(x)

    return x

class MMFuser(nn.Module):
    def __init__(self, embed_dim=1024):
        super(MMFuser, self).__init__()
        self.cross_attention = Attention(embed_dim)
        self.self_attention = Attention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.gamma1 = nn.Parameter(torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(torch.ones(embed_dim)) 
        self.mlp = MultiLayerPerceptron(embed_dim, int(4.0), embed_dim)
    def forward(self, input_features, query_feature):
        """
        Parameters:
        - input_features: Concatenation of the outputs of the (n-1) blocks of the vision encoder (Shape: [batch_size, nb_tokens*n-1, embed_dim])
        - query_feature: Output of the final (n-th) block of the vision encoder (Shape: [batch_size, nb_tokens, embed_dim])
        
        Returns:
        - Visual Feature: Final fused features after cross-attention and self-attention.
        """


        norm_X = self.norm1(input_features)
        norm_F_L  = self.norm2(query_feature)

        cross_attended = self.mlp(self.cross_attention(norm_F_L, norm_X, norm_X))

        norm_cross_attended = self.norm3(cross_attended)
        self_attended = self.self_attention(norm_cross_attended,norm_cross_attended,norm_cross_attended)

        skip_connect = cross_attended + self.gamma1 * self_attended

        F_visual = norm_F_L + self.gamma2 * skip_connect
         

        return F_visual

class DeeCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", layer_indices= None, lora_r=16, lora_alpha=32, lora_dropout=0.05, select_feature='patch'):
        super(DeeCLIP, self).__init__()

        # Load the CLIP model and processor
        self.layer_indices = layer_indices
        self.select_feature = select_feature
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()  # Set the model to evaluation mode

        # Configure LoRA
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                f"vision_model.encoder.layers.{i}.self_attn.q_proj" for i in range(24)] + [
                f"vision_model.encoder.layers.{i}.self_attn.k_proj" for i in range(24)]+ [
                f"vision_model.encoder.layers.{i}.self_attn.v_proj" for i in range(24)],
            bias="none"
        )
                
        # LoRA to the image encoder
        self.clip_model = get_peft_model(self.model, self.lora_config)
        self.mmfuser = MMFuser(1024)
        self.clip_model.visual_projection = nn.Linear(1024, self.model.config.projection_dim)
        torch.nn.init.xavier_uniform_(self.clip_model.visual_projection.weight)
        
        self.clip_model.fc = nn.Linear(self.model.config.projection_dim, 1)
        torch.nn.init.normal_(self.clip_model.fc.weight.data, 0.0, 0.02)

        # Ensure only the classification layer is trainable
        for param in self.clip_model.fc.parameters():
            param.requires_grad = True 
        for param in  self.clip_model.visual_projection.parameters():
            param.requires_grad = True 

        del self.model

    def feature_select(self, image_forward_outs):
        image_features = None
        if isinstance(self.layer_indices, int):
            image_features = image_forward_outs.hidden_states[self.layer_indices]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
        elif isinstance(self.layer_indices, list):
            image_features = [image_forward_outs.hidden_states[ix] for ix in self.layer_indices]  # list[tensor,...]
            if self.select_feature == 'patch':
                image_features = [features[:, 1:] for features in image_features]
                
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            image_features = torch.stack(image_features, dim=0)

        if image_features == None:
            raise ValueError(f"image features in feature_select is None...")

        return image_features


    def forward(self, images, train=True):
        
        # if type(images) is list:
        #     image_features = []
        #     for image in images:
        #         image_forward_out = self.clip_model(image.unsqueeze(0),
        #                                               output_hidden_states=True)
        #         image_feature = self.feature_select(image_forward_out).to(image.dtype)
        #         image_features.append(image_feature)
        # else:
        #     image_forward_outs = self.clip_model(images,
        #                                            output_hidden_states=True)
        #     image_features = self.feature_select(image_forward_outs).to(images.dtype)

        image_forward_outs = self.clip_model(images, output_hidden_states=True)

        images_features = self.feature_select(image_forward_outs)
        
        images_features = [images_features[i] for i in range(images_features.size(0))]

        # print(torch.cat(images_features[:-1], dim=1).size())
        visual_feature = self.mmfuser( torch.cat(images_features[:-1], dim=1), images_features[-1])

        Vprojected_features = self.clip_model.visual_projection(visual_feature)

        image_embeds = Vprojected_features / Vprojected_features.norm(p=2, dim=-1, keepdim=True)




        if train:
            return image_embeds, self.clip_model.fc(image_embeds.mean(dim=1))
    
        return  images_features, visual_feature, self.clip_model.fc(image_embeds.mean(dim=1))

# if __name__ == "__main__":
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     deeclip = DeeCLIP(layer_indices=[1, 3, 5, 8, 10, 13, 15, 17, 19, 21, 22, 23]).to(device)

#     loss_fn = TripletLoss()
#     image_tensor = torch.rand(2, 3, 224, 224).to(device)  


#     outputs = deeclip(image_tensor, False)
#     loss = loss_fn(outputs,outputs,outputs)

#     print(loss)


