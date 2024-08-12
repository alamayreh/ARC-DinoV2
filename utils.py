import os
import cv2
import json
import torch
import random
import numpy as np

from addict import Dict
from matplotlib import colors
from torchvision import transforms

class DictDefault(Dict):
    """
    A Dict that returns None instead of returning empty Dict for missing keys.
    """

    def __missing__(self, key):
        return None

    def __or__(self, other):
        return DictDefault(super().__or__(other))

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def set_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def image_to_rgb(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    x = std * x + mean
    x = np.clip(x, 0, 1)

    return x

def pad_to_30x30(grid):
    # Convert the grid to a numpy array for easier manipulation
    grid = np.array(grid)
    h, w = grid.shape

    # Pad the grid to make it 30x30
    pad_h = 30 - h
    pad_w = 30 - w

    padded_grid = np.pad(
        grid,
        pad_width=((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=10 
    )
        
    # Convert to tensor
    padded_grid = torch.tensor(padded_grid, dtype=torch.long)  # Shape: [30, 30]
        
    return padded_grid

def scale_and_pad(grid, target_size, scale_factor, cmap, padding_color):
    # Rescale and pad the grid to the target size (224x224)
    grid = np.array(grid)

    scaled_grid = np.kron(grid, np.ones((scale_factor, scale_factor)))
        
    pad_h = (target_size - scaled_grid.shape[0]) // 2
    pad_w = (target_size - scaled_grid.shape[1]) // 2
        
    # Apply padding with padding color
    padded_grid = cv2.copyMakeBorder(
        scaled_grid,
        top=pad_h,
        bottom=target_size - scaled_grid.shape[0] - pad_h,
        left=pad_w,
        right=target_size - scaled_grid.shape[1] - pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color  # 10 corresponds to white (#FFFFFF)
    )
        
    # Map each value to an RGB color using the colormap
    rgb_image = np.zeros((padded_grid.shape[0], padded_grid.shape[1], 3), dtype=np.float32)
    for i in range(11):  # Since we have 11 unique values (0-10)
        rgb_image[padded_grid == i] = colors.hex2color(cmap.colors[i])
        
    # Convert the RGB values from [0,1] to [0,255] for visualization (if needed)
    rgb_image = (rgb_image * 255).astype(np.uint8)
       
    # Define the transformation: ToTensor and Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [3, 224, 224] and scales values to [0, 1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize with ImageNet values
    ])

    transformed_grid = transform(rgb_image)  # Shape: [3, 224, 224]
        
    return transformed_grid

def remove_padding(padded_grid, pad_value=10):
    padded_grid = np.array(padded_grid)
    rows_to_keep = np.any(padded_grid != pad_value, axis=1)
    cols_to_keep = np.any(padded_grid != pad_value, axis=0)
    trimmed_grid = padded_grid[rows_to_keep][:, cols_to_keep]
    trimmed_grid = torch.tensor(trimmed_grid, dtype=torch.long)
    
    return trimmed_grid