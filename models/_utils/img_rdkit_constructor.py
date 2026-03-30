import os
import math

# set huggingface dir and web mirror
os.environ['HF_HOME'] = "../../_soft/HF_HOME"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import tempfile
import torch
from torch import nn
import numpy as np
from PIL import Image
from mbapy.file import opts_file
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit import Chem
from rdkit.Chem import Draw
import timm
import torchvision
import torchvision.transforms as transforms
import io
import subprocess

class MaskMol(nn.Module):
    def __init__(self, pretrained=True):
        super(MaskMol, self).__init__()

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        # Remove the classification head to get features only
        self.vit.head = nn.Identity()
        
        # self.atom_patch_classifier = nn.Linear(768, 10)
        # self.bond_patch_classifier = nn.Linear(768, 4)
        # self.motif_classifier = nn.Linear(768, 200)

        self.regressor = nn.Linear(768, 1)

    def forward(self, x):
        x = self.vit(x)
#         x = torch.relu(x)
        x = self.regressor(x)

        return x


class RdkitViT(nn.Module):
    """RdkitViT class for processing molecular images with timm pre-trained Vision Transformer models."""
    def __init__(self, resolution=224, pretrained=True):
        super(RdkitViT, self).__init__()
        
        # Select the appropriate model based on resolution
        if resolution == 224:
            model_name = "vit_base_patch16_224"
        elif resolution == 512:
            # Use a valid 512 resolution model from timm library
            model_name = "vit_base_patch16_siglip_512"
        else:
            raise ValueError(f"Unsupported resolution: {resolution}. Supported resolutions are 224 and 512.")
        
        # Load the pre-trained ViT model without classification head
        self.vit = timm.create_model(model_name, pretrained=pretrained,
                                     cache_dir="../../_soft/HF_HOME/hub", num_classes=0)
        # Ensure the model is in evaluation mode
        self.vit.eval()
        
        # Store the resolution for reference
        self.resolution = resolution
    
    def forward(self, x):
        """Forward pass through the ViT model to get features."""
        return self.vit(x)
    

class ImageMol(nn.Module):
    """http://dx.doi.org/10.1038/s42256-022-00557-6
    https://github.com/ChengF-Lab/ImageMol"""
    def __init__(self, baseModel, jigsaw_classes, label1_classes, label2_classes, label3_classes):
        super(ImageMol, self).__init__()

        self.embedding_layer = nn.Sequential(*list(baseModel.children())[:-1])

        self.bn = nn.BatchNorm1d(512)

        self.jigsaw_classifier = nn.Linear(512, jigsaw_classes)
        self.class_classifier1 = nn.Linear(512, label1_classes)
        self.class_classifier2 = nn.Linear(512, label2_classes)
        self.class_classifier3 = nn.Linear(512, label3_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1) # [N, 512]

        x1 = self.jigsaw_classifier(x) # [N, 101]
        x2 = self.class_classifier1(x) # [N, 100]
        x3 = self.class_classifier2(x).reshape(x.size(0), 10, 100).mean(dim=1) # [N, 100]
        x4 = self.class_classifier3(x).reshape(x.size(0), 100, 100).mean(dim=1) # [N, 100]

        return torch.cat([x, x1, x2, x3, x4], dim=1) # [1, 512+101+100+100+100]


def load_image_model(model_name="MaskMol", resolution=224):
    """Load molecular image model.
    
    Args:
        model_name: Model type to load, either "MaskMol" or "rdkit_vit"
        resolution: Target image resolution (224 or 512) for rdkit_vit model
        
    Returns:
        Loaded image model
    """
    if model_name == "MaskMol":
        # Original MaskMol loading logic
        model_path = '../EHIGN_dataset/_pretrained/MaskMol/MaskMol_base.pth.tar'
        model = MaskMol(pretrained=False)
        # load into cpu
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        del_keys = ['module.atom_patch_classifier.weight', 'module.atom_patch_classifier.bias', 'module.bond_patch_classifier.weight',
            'module.bond_patch_classifier.bias', 'module.motif_classifier.weight', 'module.motif_classifier.bias'] 
        for k in del_keys:
            del checkpoint['state_dict'][k]
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}, strict=False)
    elif model_name == "ImageMol":
        jigsaw_classes = 100 + 1
        label1_classes = 100
        label2_classes = 1000
        label3_classes = 10000
        res18 = torchvision.models.resnet18(pretrained=False)
        res18.fc = torch.nn.Linear(res18.fc.in_features, 2)
        model = ImageMol(res18, jigsaw_classes, label1_classes, label2_classes, label3_classes)
        weights = torch.load('../EHIGN_dataset/_pretrained/ImageMol.pth.tar')["state_dict"]
        model.load_state_dict(weights, strict=True)
    elif model_name == "rdkit_vit":
        # New RdkitViT model loading
        model = RdkitViT(resolution=resolution, pretrained=True)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported models are 'MaskMol' and 'rdkit_vit'.")
    
    model.eval()
    return model


def get_img_transformer(resolution=224):
    """get image transformer with customizable resolution
    
    Args:
        resolution: Target image resolution for processing
    
    Returns:
        Transformer composed of center crop, to tensor, and normalization
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    img_transformer_test = [transforms.CenterCrop(resolution), transforms.ToTensor(), normalize]
    img_transformer=transforms.Compose(img_transformer_test)
    return img_transformer


def smiles_to_hidden_state(smiles, model, img_transformer, resolution=224):
    """convert smiles to hidden state with rotation augmentation
    
    Args:
        smiles: SMILES string of the molecule
        model: MaskMol model
        img_transformer: Image transformer
        resolution: Target image resolution
    
    Returns:
        Hidden states of shape [4, 768] representing the four rotated views
    """
    lig_mol = Chem.MolFromSmiles(smiles)
    # Generate image with the specified resolution
    img = Draw.MolsToGridImage([lig_mol], molsPerRow=1, subImgSize=(resolution, resolution))
    img = Image.frombytes("RGB", img.size, img.tobytes())
    
    # Generate four rotated versions of the image (0°, 90°, 180°, 270°)
    img_tensors = []
    for angle in [0, 90, 180, 270]:
        rotated_img = img.rotate(angle, expand=1)
        # Ensure the image is at the specified resolution after rotation
        if rotated_img.size != (resolution, resolution):
            # Center crop to the specified resolution if needed
            transform = transforms.Compose([transforms.CenterCrop(resolution)])
            rotated_img = transform(rotated_img)
        # Apply the transformer (to tensor and normalize)
        img_tensor = img_transformer(rotated_img)
        img_tensors.append(img_tensor)
    
    # Create a batch of the four rotated images
    batch = torch.stack(img_tensors)
    
    # Get hidden states for the entire batch
    if hasattr(model, "vit"):
        model_device = model.vit.blocks[0].attn.qkv.weight.device
        hidden_states = model.vit(batch.to(model_device))
    else:
        model_device = model.embedding_layer[0].weight.device
        hidden_states = model(batch.to(model_device))
    
    return hidden_states



if __name__ == "__main__":
    # Example SMILES for testing
    lig_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    print("Example 1: Using MaskMol model")
    # Load MaskMol model (default)
    model_maskmol = load_image_model(model_name="MaskMol")
    # Get transformer for MaskMol (224x224 is required for MaskMol)
    img_transformer_maskmol = get_img_transformer(resolution=224)
    # Convert SMILES to hidden state with rotation augmentation
    hidden_state_maskmol = smiles_to_hidden_state(lig_smiles, model_maskmol, img_transformer_maskmol, resolution=224)
    print(f"Hidden state shape with MaskMol: {hidden_state_maskmol.shape}")
    
    print("\nExample 2: Using RdkitViT with 224x224 resolution")
    # Load RdkitViT with 224x224 resolution
    model_vit_224 = load_image_model(model_name="rdkit_vit", resolution=224)
    # Get transformer for 224x224 resolution
    img_transformer_vit_224 = get_img_transformer(resolution=224)
    # Convert SMILES to hidden state with rotation augmentation
    hidden_state_vit_224 = smiles_to_hidden_state(lig_smiles, model_vit_224, img_transformer_vit_224, resolution=224)
    print(f"Hidden state shape with RdkitViT (224x224): {hidden_state_vit_224.shape}")
    
    print("\nExample 3: Using RdkitViT with 512x512 resolution")
    # Load RdkitViT with 512x512 resolution
    model_vit_512 = load_image_model(model_name="rdkit_vit", resolution=512)
    # Get transformer for 512x512 resolution
    img_transformer_vit_512 = get_img_transformer(resolution=512)
    # Convert SMILES to hidden state with rotation augmentation
    hidden_state_vit_512 = smiles_to_hidden_state(lig_smiles, model_vit_512, img_transformer_vit_512, resolution=512)
    print(f"Hidden state shape with RdkitViT (512x512): {hidden_state_vit_512.shape}")
    
    print("\nExample 4: Using ImageMol model")
    # Load ImageMol model
    model_imagemol = load_image_model(model_name="ImageMol")
    # Get transformer for ImageMol (224x224 is required for ImageMol)
    img_transformer_imagemol = get_img_transformer(resolution=224)
    # Convert SMILES to hidden state with rotation augmentation
    hidden_state_imagemol = smiles_to_hidden_state(lig_smiles, model_imagemol, img_transformer_imagemol, resolution=224)
    print(f"Hidden state shape with ImageMol: {hidden_state_imagemol.shape}")