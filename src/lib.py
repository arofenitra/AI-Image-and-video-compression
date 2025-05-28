import glob
import pandas as pd
import cv2
import gc
import numpy as np
import random
import imageio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from urllib.request import urlopen
from tensorflow_docs.vis import embed
import os,  argparse, math, itertools
import torch, torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from compressai.zoo import bmshj2018_factorized, ssf2020
import torch
import av
import numpy as np
from compressai.zoo import ssf2020
import os 
import sys 
import os, math, itertools, shutil, gc
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from torchvision import transforms
from compressai.zoo import bmshj2018_factorized, ssf2020 
import numpy as _np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from compressai.zoo import models 
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import torch
from kornia.losses import PSNRLoss, SSIMLoss
from torchmetrics.image import VisualInformationFidelity
import torch.nn.functional as F
from compressai.zoo import models
from torchvision import transforms
import PIL 
import numpy as np 
from skimage import io , img_as_float
# from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import gc 
import kornia
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from IPython.display import display
import os
import scipy.io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from kornia.losses import SSIMLoss, PSNRLoss
from torchmetrics.image import VisualInformationFidelity


def pad_to_multiple(x, m):
    """
    Reflect-pad so (H, W) is a multiple of m.
    SSF / other video codecs need m = 128; most image codecs work with 64.
    """
    B, C, H, W = x.shape
    Hp, Wp = (m - H % m) % m, (m - W % m) % m
    return F.pad(x, (0, Wp, 0, Hp), mode="reflect"), (H, W)

def bits_in(strings):
    return sum(len(s) * 8 for s in flatten(strings))

def flatten(l):
    for el in l:
        if isinstance(el, (list, tuple)):
            yield from flatten(el)
        else:
            yield el
def classify_video_url(model, video_url, n_frames=10):
    # Download video from URL
    video = urlopen(video_url)
    with open('temp_video.avi', 'wb') as f:
        f.write(video.read())

    # Create frames from the downloaded video
    video_frames = frames_from_video_file('temp_video.avi', n_frames=n_frames)

    # Predict using the model
    predictions = model.predict(np.expand_dims(video_frames, axis=0))

    # Classify the video based on predictions
    predicted_class = np.argmax(predictions)
    predicted_class_name = CFG.classes[predicted_class]

    print(f"Predicted Class: {predicted_class_name}")
    return predicted_class_name
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(len(class_names), len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=10)
  return embed.embed_file('./animation.gif')


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        # Load the pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        # Extract the features from the VGG16 model
        self.features = vgg16.features

        # Define the side layers
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)

        # Define the fuse layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

        # Freeze the VGG16 layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract the feature maps
        feat1 = self.features[:4](x)
        feat2 = self.features[4:9](feat1)
        feat3 = self.features[9:16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:30](feat4)

        # Apply the side layers
        side1 = self.side1(feat1)
        side2 = self.side2(feat2)
        side3 = self.side3(feat3)
        side4 = self.side4(feat4)
        side5 = self.side5(feat5)

        # Upsample the side outputs to the same size
        side2 = nn.functional.interpolate(side2, size=side1.size()[2:], mode='bilinear', align_corners=True)
        side3 = nn.functional.interpolate(side3, size=side1.size()[2:], mode='bilinear', align_corners=True)
        side4 = nn.functional.interpolate(side4, size=side1.size()[2:], mode='bilinear', align_corners=True)
        side5 = nn.functional.interpolate(side5, size=side1.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate the side outputs
        fused = torch.cat((side1, side2, side3, side4, side5), dim=1)

        # Apply the fuse layer
        output = self.fuse(fused)

        # Apply sigmoid activation
        output = torch.sigmoid(output)

        return output

class EnhancedHED(nn.Module):
    """Enhanced HED model with better architecture"""
    def __init__(self, pretrained=True):
        super(EnhancedHED, self).__init__()
        
        # Load VGG16 backbone
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        
        # Side output layers with proper initialization
        self.side1 = nn.Sequential(
            nn.Conv2d(64, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side2 = nn.Sequential(
            nn.Conv2d(128, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side3 = nn.Sequential(
            nn.Conv2d(256, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side4 = nn.Sequential(
            nn.Conv2d(512, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side5 = nn.Sequential(
            nn.Conv2d(512, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        
        # Fusion layer
        self.fuse = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Optionally freeze VGG layers
        # for param in self.features.parameters():
        #     param.requires_grad = False
    
    def _initialize_weights(self):
        for module in [self.side1, self.side2, self.side3, self.side4, self.side5, self.fuse]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h, w = x.size()[2:]
        
        # Extract features at different scales
        feat1 = self.features[:4](x)      # 64 channels
        feat2 = self.features[4:9](feat1)  # 128 channels
        feat3 = self.features[9:16](feat2) # 256 channels
        feat4 = self.features[16:23](feat3) # 512 channels
        feat5 = self.features[23:30](feat4) # 512 channels
        
        # Side outputs
        side1 = self.side1(feat1)
        side2 = self.side2(feat2)
        side3 = self.side3(feat3)
        side4 = self.side4(feat4)
        side5 = self.side5(feat5)
        
        # Upsample to original size
        side1 = F.interpolate(side1, size=(h, w), mode='bilinear', align_corners=True)
        side2 = F.interpolate(side2, size=(h, w), mode='bilinear', align_corners=True)
        side3 = F.interpolate(side3, size=(h, w), mode='bilinear', align_corners=True)
        side4 = F.interpolate(side4, size=(h, w), mode='bilinear', align_corners=True)
        side5 = F.interpolate(side5, size=(h, w), mode='bilinear', align_corners=True)
        
        # Fusion
        fused = torch.cat([side1, side2, side3, side4, side5], dim=1)
        output = self.fuse(fused)
        
        # Apply sigmoid
        output = torch.sigmoid(output)
        
        return output, [torch.sigmoid(side1), torch.sigmoid(side2), 
                       torch.sigmoid(side3), torch.sigmoid(side4), torch.sigmoid(side5)]

class BalancedBCELoss(nn.Module):
    """Balanced Binary Cross Entropy Loss for edge detection"""
    def __init__(self, pos_weight=None):
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        if self.pos_weight is None:
            # Calculate positive weight dynamically
            pos_pixels = torch.sum(target)
            neg_pixels = torch.sum(1 - target)
            if pos_pixels > 0:
                pos_weight = neg_pixels / pos_pixels
            else:
                pos_weight = 1.0
        else:
            pos_weight = self.pos_weight
            
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        loss = -pos_weight * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return torch.mean(loss)

class DiceLoss(nn.Module):
    """Dice Loss for edge detection"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = torch.sum(pred_flat * target_flat)
        dice = (2.0 * intersection + self.smooth) / (torch.sum(pred_flat) + torch.sum(target_flat) + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined loss: Balanced BCE + Dice Loss"""
    def __init__(self, bce_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.bce_loss = BalancedBCELoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

class EnhancedHED(nn.Module):
    """Enhanced HED model with better architecture"""
    def __init__(self, pretrained=True):
        super(EnhancedHED, self).__init__()
        
        # Load VGG16 backbone
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        
        # Side output layers with proper initialization
        self.side1 = nn.Sequential(
            nn.Conv2d(64, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side2 = nn.Sequential(
            nn.Conv2d(128, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side3 = nn.Sequential(
            nn.Conv2d(256, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side4 = nn.Sequential(
            nn.Conv2d(512, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        self.side5 = nn.Sequential(
            nn.Conv2d(512, 21, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(21, 1, kernel_size=1)
        )
        
        # Fusion layer
        self.fuse = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Optionally freeze VGG layers
        # for param in self.features.parameters():
        #     param.requires_grad = False
    
    def _initialize_weights(self):
        for module in [self.side1, self.side2, self.side3, self.side4, self.side5, self.fuse]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h, w = x.size()[2:]
        
        # Extract features at different scales
        feat1 = self.features[:4](x)      # 64 channels
        feat2 = self.features[4:9](feat1)  # 128 channels
        feat3 = self.features[9:16](feat2) # 256 channels
        feat4 = self.features[16:23](feat3) # 512 channels
        feat5 = self.features[23:30](feat4) # 512 channels
        
        # Side outputs
        side1 = self.side1(feat1)
        side2 = self.side2(feat2)
        side3 = self.side3(feat3)
        side4 = self.side4(feat4)
        side5 = self.side5(feat5)
        
        # Upsample to original size
        side1 = F.interpolate(side1, size=(h, w), mode='bilinear', align_corners=True)
        side2 = F.interpolate(side2, size=(h, w), mode='bilinear', align_corners=True)
        side3 = F.interpolate(side3, size=(h, w), mode='bilinear', align_corners=True)
        side4 = F.interpolate(side4, size=(h, w), mode='bilinear', align_corners=True)
        side5 = F.interpolate(side5, size=(h, w), mode='bilinear', align_corners=True)
        
        # Fusion
        fused = torch.cat([side1, side2, side3, side4, side5], dim=1)
        output = self.fuse(fused)
        
        # Apply sigmoid
        output = torch.sigmoid(output)
        
        return output, [torch.sigmoid(side1), torch.sigmoid(side2), 
                       torch.sigmoid(side3), torch.sigmoid(side4), torch.sigmoid(side5)]

class RCF(nn.Module):
    """Richer Convolutional Features for Edge Detection"""
    def __init__(self, pretrained=True):
        super(RCF, self).__init__()
        
        # VGG16 backbone
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        
        # RCF blocks for multi-scale feature extraction
        self.rcf_block1 = self._make_rcf_block(64, [1, 2, 3])
        self.rcf_block2 = self._make_rcf_block(128, [1, 2, 3])
        self.rcf_block3 = self._make_rcf_block(256, [1, 2, 3, 5])
        self.rcf_block4 = self._make_rcf_block(512, [1, 2, 3, 5])
        self.rcf_block5 = self._make_rcf_block(512, [1, 2, 3, 5])
        
        # Side outputs
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)
        
        # Fusion
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
        
        self._initialize_weights()
    
    def _make_rcf_block(self, in_channels, dilations):
        layers = []
        for dilation in dilations:
            layers.append(nn.Conv2d(in_channels, in_channels, 
                                  kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h, w = x.size()[2:]
        
        # Extract features
        feat1 = self.features[:4](x)
        feat2 = self.features[4:9](feat1)
        feat3 = self.features[9:16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:30](feat4)
        
        # Apply RCF blocks
        feat1 = self.rcf_block1(feat1)
        feat2 = self.rcf_block2(feat2)
        feat3 = self.rcf_block3(feat3)
        feat4 = self.rcf_block4(feat4)
        feat5 = self.rcf_block5(feat5)
        
        # Side outputs
        side1 = self.side1(feat1)
        side2 = self.side2(feat2)
        side3 = self.side3(feat3)
        side4 = self.side4(feat4)
        side5 = self.side5(feat5)
        
        # Upsample
        side1 = F.interpolate(side1, size=(h, w), mode='bilinear', align_corners=True)
        side2 = F.interpolate(side2, size=(h, w), mode='bilinear', align_corners=True)
        side3 = F.interpolate(side3, size=(h, w), mode='bilinear', align_corners=True)
        side4 = F.interpolate(side4, size=(h, w), mode='bilinear', align_corners=True)
        side5 = F.interpolate(side5, size=(h, w), mode='bilinear', align_corners=True)
        
        # Fusion
        fused = torch.cat([side1, side2, side3, side4, side5], dim=1)
        output = self.fuse(fused)
        
        return torch.sigmoid(output)

class ImprovedEdgeDetectionDataset(Dataset):
    """Improved dataset with better error handling and preprocessing"""
    def __init__(self, image_paths, edge_paths, transform=None, augment=False):
        # Filter and match files properly
        self.image_paths, self.edge_paths = self._match_files(image_paths, edge_paths)
        self.transform = transform
        self.augment = augment
        
        # Data augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
        # Debug: Print structure of first .mat file
        if len(self.edge_paths) > 0:
            self._debug_mat_structure(self.edge_paths[0])
    
    def _debug_mat_structure(self, mat_path):
        """Debug function to print .mat file structure"""
        try:
            mat_data = scipy.io.loadmat(mat_path)
            print(f"\n=== Debugging .mat file structure for: {mat_path} ===")
            print(f"Keys in .mat file: {list(mat_data.keys())}")
            
            if 'groundTruth' in mat_data:
                gt = mat_data['groundTruth']
                print(f"groundTruth shape: {gt.shape}")
                print(f"groundTruth type: {type(gt)}")
                print(f"groundTruth dtype: {gt.dtype}")
                
                if gt.size > 0:
                    first_element = gt[0, 0]
                    print(f"First element type: {type(first_element)}")
                    
                    # Check if it's a structured array
                    if hasattr(first_element, 'dtype'):
                        print(f"First element dtype: {first_element.dtype}")
                        if hasattr(first_element.dtype, 'names') and first_element.dtype.names:
                            print(f"Field names: {first_element.dtype.names}")
                            for field in first_element.dtype.names:
                                field_data = first_element[field]
                                print(f"  {field}: shape={getattr(field_data, 'shape', 'N/A')}, "
                                      f"type={type(field_data)}")
                                if hasattr(field_data, 'shape') and len(field_data.shape) > 0:
                                    print(f"    First few values: {field_data.flat[:5] if field_data.size > 0 else 'Empty'}")
                    
                    # Try different access patterns
                    try:
                        boundaries = first_element['Boundaries'][0, 0]
                        print(f"Boundaries shape: {boundaries.shape}")
                        print(f"Boundaries dtype: {boundaries.dtype}")
                        print(f"Boundaries min/max: {boundaries.min()}/{boundaries.max()}")
                        print(f"Unique values in Boundaries: {np.unique(boundaries)}")
                    except:
                        print("Could not access Boundaries field")
                        
                    try:
                        segmentation = first_element['Segmentation'][0, 0]
                        print(f"Segmentation shape: {segmentation.shape}")
                    except:
                        print("Could not access Segmentation field")
            
            print("=== End debug ===\n")
        except Exception as e:
            print(f"Error debugging .mat file: {e}")
    
    def _match_files(self, image_paths, edge_paths):
        """Match image and edge files by filename"""
        image_dict = {}
        edge_dict = {}
        
        # Create dictionaries with base filenames
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_dict[base_name] = img_path
        
        for edge_path in edge_paths:
            base_name = os.path.splitext(os.path.basename(edge_path))[0]
            edge_dict[base_name] = edge_path
        
        # Find matching pairs
        matched_images = []
        matched_edges = []
        
        for base_name in image_dict.keys():
            if base_name in edge_dict:
                matched_images.append(image_dict[base_name])
                matched_edges.append(edge_dict[base_name])
        
        print(f"Matched {len(matched_images)} image-edge pairs")
        return matched_images, matched_edges
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load image
            image = Image.open(self.image_paths[idx]).convert('RGB')
            
            # Load edge map from .mat file
            mat_data = scipy.io.loadmat(self.edge_paths[idx])
            
            # Handle BSDS500 .mat file structure
            if 'groundTruth' in mat_data:
                gt_data = mat_data['groundTruth']
                first_element = gt_data[0, 0]
                
                # Access the Boundaries field
                if hasattr(first_element.dtype, 'names') and 'Boundaries' in first_element.dtype.names:
                    edge_map = first_element['Boundaries'][0, 0]
                else:
                    # Fallback access pattern
                    edge_map = gt_data[0, 0][0][0][1]
            else:
                # Try direct access for other formats
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                edge_map = mat_data[keys[0]]
            
            # Debug print for first few samples
            if idx < 3:
                print(f"Sample {idx}: Edge map shape: {edge_map.shape}, "
                      f"dtype: {edge_map.dtype}, min/max: {edge_map.min()}/{edge_map.max()}")
            
            # Ensure binary values (0 or 1)
            edge_map = (edge_map > 0).astype(np.float32)
            
            # Convert to PIL Image
            edge_map_pil = Image.fromarray((edge_map * 255).astype(np.uint8), mode='L')
            
            # Apply augmentation if specified
            if self.augment and hasattr(self, 'aug_transform'):
                # Apply same transform to both image and edge map
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                image = self.aug_transform(image)
                torch.manual_seed(seed)
                edge_map_pil = self.aug_transform(edge_map_pil)
            
            # Apply main transforms
            if self.transform:
                image = self.transform(image)
                edge_map_tensor = self.transform(edge_map_pil)
            else:
                # Convert to tensor manually if no transform
                image = transforms.ToTensor()(image)
                edge_map_tensor = transforms.ToTensor()(edge_map_pil)
                
            # Ensure edge map is single channel and binary
            if edge_map_tensor.shape[0] == 3:
                edge_map_tensor = edge_map_tensor[0:1]  # Take first channel
            
            # Ensure binary values (0 or 1) in tensor
            edge_map_tensor = (edge_map_tensor > 0.5).float()
            
            # Debug print for first few samples
            if idx < 3:
                print(f"Sample {idx}: Final tensor shapes - Image: {image.shape}, "
                      f"Edge: {edge_map_tensor.shape}")
                print(f"Edge tensor min/max: {edge_map_tensor.min()}/{edge_map_tensor.max()}")
                print(f"Edge tensor unique values: {torch.unique(edge_map_tensor)}")
            
            return image, edge_map_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} from {self.edge_paths[idx]}: {e}")
            # Return a dummy sample with proper binary values
            if self.transform:
                dummy_img = torch.zeros(3, 320, 320)  # Match transform size
                dummy_edge = torch.zeros(1, 320, 320)
            else:
                dummy_img = torch.zeros(3, 256, 256)
                dummy_edge = torch.zeros(1, 256, 256)
            return dummy_img, dummy_edge

def calculate_edge_metrics(predictions, targets, threshold=0.5):
    """Calculate comprehensive metrics for edge detection"""
    # Debug prints
    print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
    print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
    print(f"Predictions min/max: {predictions.min():.4f}/{predictions.max():.4f}")
    print(f"Targets min/max: {targets.min():.4f}/{targets.max():.4f}")
    print(f"Predictions unique values: {torch.unique(predictions)[:10]}")  # Show first 10
    print(f"Targets unique values: {torch.unique(targets)}")
    
    # Ensure predictions are in [0, 1] range and apply threshold
    predictions = torch.clamp(predictions, 0, 1)
    pred_binary = (predictions > threshold).float()
    
    # Ensure targets are binary (0 or 1)
    targets_binary = (targets > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy().astype(int)
    target_flat = targets_binary.view(-1).cpu().numpy().astype(int)
    
    # Debug prints for flattened arrays
    print(f"Flattened predictions shape: {pred_flat.shape}, dtype: {pred_flat.dtype}")
    print(f"Flattened targets shape: {target_flat.shape}, dtype: {target_flat.dtype}")
    print(f"Unique pred_flat values: {np.unique(pred_flat)}")
    print(f"Unique target_flat values: {np.unique(target_flat)}")
    
    # Calculate metrics with proper error handling
    try:
        accuracy = accuracy_score(target_flat, pred_flat)
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'edge_ratio': 0.0
        }
    
    # Edge-specific metrics
    edge_pixels = np.sum(target_flat)
    total_pixels = len(target_flat)
    edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'edge_ratio': edge_ratio
    }

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Enhanced training function"""
    
    # Loss function and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, edges) in enumerate(train_loader):
            images, edges = images.to(device), edges.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, EnhancedHED):
                outputs, side_outputs = model(images)
                loss = criterion(outputs, edges)
                # Add loss from side outputs
                for side_out in side_outputs:
                    loss += 0.5 * criterion(side_out, edges)
            else:
                outputs = model(images)
                loss = criterion(outputs, edges)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, edges in val_loader:
                images, edges = images.to(device), edges.to(device)
                
                if isinstance(model, EnhancedHED):
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, edges)
                val_loss += loss.item()
                
                all_predictions.append(outputs.cpu())
                all_targets.append(edges.cpu())
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_edge_metrics(all_predictions, all_targets)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  Metrics - Acc: {metrics["accuracy"]:.4f}, Prec: {metrics["precision"]:.4f}, '
              f'Rec: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}')
        print(f'  Edge Ratio: {metrics["edge_ratio"]:.4f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_edge_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses
def psnr(x,y,max_val = 1.0,device = device):
    psnr = PSNRLoss(max_val=1.0)
    if x.device ==device:
        return psnr(x,y).item()
    else:
        x = x.to(device)
        y = y.to(device)
        return psnr(x,y).item()

def ssim(x,y, window_size=11,device=device):
    ssim_loss = SSIMLoss(window_size=window_size)
    if device==x.device:
        return ssim_loss(x,y).item()
    else:
        x = x.to(device)
        y = y.to(device)
        return ssim_loss(x,y).item()


def VIF(x,y,device=device):
    VIF = VisualInformationFidelity().to(device)
    if device==x.device:
        return VIF(x,y).item()
    else:
        x = x.to(device)
        y = y.to(device)
        return VIF(x,y).item()
        
# Function to compute bpp from compressed strings
def compute_bpp_from_compressed(strings_list, num_pixels):
    # Calculate the total number of bits
    total_bits = sum(len(s) for s in strings_list) * 8  # Each byte is 8 bits

    # Compute bpp
    bpp = total_bits / num_pixels

    return bpp
  
def pad_to_multiple(img, multiple=128):
    w, h = img.size
    new_w = math.ceil(w / multiple) * multiple
    new_h = math.ceil(h / multiple) * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
    return transforms.functional.pad(img, padding, fill=0)

transform = transforms.Compose([
    transforms.Lambda(pad_to_multiple),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])
# Function to get file paths
def get_file_paths(base_dir, subdirs, extension):
    file_paths = []
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path):
            files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(extension)]
            file_paths.extend(files)
    return file_paths

class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomDenseNet, self).__init__()
        self.densenet = models.densenet201(pretrained=True)
        
        # Remove the classification head
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        
        # Define new classification head
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(11520, 512)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.densenet(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
def numpy_to_torchloader(x, y, batch_size):
    tensor_x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to PyTorch format (N, C, H, W)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
def evaluate(loader, dataset_name="Test"):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"{dataset_name}: Accuracy = {acc:.4f}")
    return np.array(all_labels), np.array(all_preds)
vif = VisualInformationFidelity()
def PSNR(x,y,max_val):
    psnr = PSNRLoss(max_val=max_val)
    return -psnr(x,y).item()
def SSIM(x,y):
    ssim_loss = SSIMLoss(window_size=11, reduction='mean')
    ssim_value = 1-ssim_loss(x,y).item()
    return ssim_value

def BPP(image_tensor,model, num_pixels):
    output = model(image_tensor)
    # Calculate BPP
    bpp = sum(
        torch.log(likelihoods).sum() / (-torch.log(2) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )
    return bpp.item()

def VIF(x,y):
    return vif(x,y).item()

def mse(x,y):
    return F.mse_loss(x,y).item()
def mae(x,y):
    return F.l1_loss(x,y).item()
def calculate_metrics(x,y,max_val,model,num_pixels):
    num_pixels = x.shape[2]*x.shape[3]
    output = model(x)
    bpp = sum(
        torch.log(likelihoods).sum() / (-torch.log(torch.tensor(2)) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )
    bpp = bpp.item()
    return {"psnr":PSNR(x,y,max_val),
            "ssim":SSIM(x,y),
            "vif":VIF(x,y),
            "mse":mse(x,y),
            "mae":mae(x,y),
            "bpp":bpp}
def compression_image_tensor(x,model,device,quality):
    x = x.to(device)
    with torch.no_grad():
        compressed = model(x)
        decompressed = compressed["x_hat"]
    return decompressed.detach()
def parse_metrics(text):
    # Create a list to store the data
    data = []
    
    # Extract model and quality from the first two lines
    lines = text.strip().split('\n')
    model_match = re.search(r'Model: ([\w-]+)', lines[0])
    quality_match = re.search(r'Quality: (\d+)', lines[1])
    
    if not model_match or not quality_match:
        raise ValueError("Model or Quality information not found in the text")
    
    model = model_match.group(1)
    quality = int(quality_match.group(1))
    
    # Parse each kodak image and its metrics
    pattern = r'kodak image (\d+)\s+\n\s*{\'psnr\': ([\d.]+), \'ssim\': ([\d.]+), \'vif\': ([\d.]+), \'mse\': ([\d.]+), \'mae\': ([\d.]+), \'bpp\': ([\d.]+)}'
    
    matches = re.finditer(pattern, text)
    
    for match in matches:
        kodak_id = int(match.group(1))
        psnr = round(float(match.group(2)), 2)
        ssim = round(float(match.group(3)), 4)
        vif = round(float(match.group(4)), 4)
        mse = float(match.group(5))
        mae = float(match.group(6))
        bpp = round(float(match.group(7)), 4)
        
        # Format MSE and MAE as scientific notation
        def format_scientific(value):
            power = math.floor(math.log10(value)) if value > 0 else 0
            mantissa = value / (10 ** power)
            mantissa_rounded = round(mantissa, 2)
            return f"{mantissa_rounded} × 10^{power}"
        
        mse_formatted = format_scientific(mse)
        mae_formatted = format_scientific(mae)
        
        # Append the data
        data.append({
            'model': model,
            'quality': quality,
            'kodak': kodak_id,
            'psnr': psnr,
            'ssim': ssim,
            'vif': vif,
            'bpp': bpp,
            'mse': mse_formatted,
            'mae': mae_formatted
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df
def compress_image(input_path, output_path, model=None):
    img = Image.open(input_path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    if model is not None:
        with th.no_grad():
            x = model(x)["x_hat"].clamp(0, 1)

    reconstructed_img = transforms.ToPILImage()(x.squeeze(0))
    reconstructed_img.save(output_path)
def train(model, device, train_loader, optimizer, epoch):
    """ Train the model for one epoch """
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")  # Progress bar

    correct = 0
    processed = 0
    running_loss = 0  # Track loss per epoch

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_pred = model(data)
        loss = criterion(y_pred, target)

        # Optional: Add L1 Regularization (if needed)
        # l1 = sum(p.abs().sum() for p in model.parameters())
        # loss += lambda1 * l1

        loss.backward()
        optimizer.step()

        # Convert loss to float before appending
        running_loss += loss.item()

        # Calculate batch accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Update progress bar
        pbar.set_description(f'Loss={loss.item():.4f} | Batch={batch_idx} | Accuracy={100*correct/processed:.2f}%')

    # Append per-epoch loss and accuracy
    train_losses.append(running_loss / len(train_loader))  # Average loss per batch
    train_acc.append(100 * correct / processed)  # Accuracy per epoch

def test(model, device, test_loader):
    """ Evaluate the model on the test set """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Accumulate loss
            test_loss += criterion(output, target).item()  # Convert tensor to float

            # Get predicted class
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Compute average loss
    test_loss /= len(test_loader)

    # Append per-epoch loss and accuracy
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_acc.append(accuracy)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return accuracy  # Return accuracy as a float

def set_parameter_requires_grad(model, feature_extracting=True):
    if feature_extracting:
        for name, param in model.named_parameters():
            param.requires_grad = False


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = pad_to_multiple(img)  # Apply padding first
    return transforms.ToTensor()(img).unsqueeze(0)
def load_model(checkpoint_path, num_classes):
    model = torch_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model.to(device)
def test_compressed(model, compressed_dir, device):
    test_data = datasets.ImageFolder(
        os.path.join(data_path, "test_compressed", compressed_dir),
        transform=test_transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"{compressed_dir} - Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy
def compress_images(plate_tensors, plate_paths, state, model_name, quality, device):
    """Compress and decompress images using a given model and quality"""
    model = models[model_name](quality=quality, pretrained=True).to(device)
    model.eval()

    # Create model-specific output directory
    model_state_dir = os.path.join(
        COMPRESSED_ROOT, f"{model_name}-q{quality}", state
    )
    os.makedirs(model_state_dir, exist_ok=True)

    for idx, tensor in enumerate(plate_tensors):
        with torch.no_grad():
            compressed = model(tensor)
            decompressed = compressed["x_hat"].clamp(0, 1)

            # Convert tensor to image
            decompressed_np = (decompressed.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(decompressed_np).save(
                os.path.join(model_state_dir, f"{os.path.splitext(os.path.basename(plate_paths[idx]))[0]}.jpg")
            )

    del model
    torch.cuda.empty_cache()
def process_state(state_dir, state, device):
    """Process all images in a state folder across different models and qualities"""
    plate_paths = [
        os.path.join(state_dir, f)
        for f in os.listdir(state_dir)
        if f.endswith((".jpg", ".png"))
    ]

    # Preprocess images
    plate_tensors = [preprocess_image(p).to(device) for p in plate_paths]

    # Compress images using all model-quality combinations
    for model_name in MODEL_NAMES:
        for quality in MODEL_QUALITIES:
            compress_images(plate_tensors, plate_paths, state, model_name, quality, device)
def load_model(checkpoint_path, num_classes):
    """Load trained ResNet50 model with custom fully connected layers."""
    model = torch_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model.to(device).eval()
def test_compressed(model, quality):
    """
    Evaluate the model on compressed datasets of a specific quality.

    Args:
        model: Trained model
        quality (int): Quality level to test (e.g., 1, 3, 6)

    Returns:
        dict: Accuracy results for each compression model at the given quality
    """
    results = {}
    for model_name in MODEL_NAMES:
        compressed_dir = f"{model_name}-q{quality}"
        test_data = datasets.ImageFolder(
            os.path.join(data_path, "test_compressed", compressed_dir),
            transform=test_transform
        )
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        results[compressed_dir] = accuracy

    return results

# 4. Evaluation Wrapper ------------------------------------------------------
def evaluate(quality):
    """
    Wrapper function to load the model and evaluate a specific quality level.

    Args:
        quality (int): Quality level to evaluate

    Returns:
        dict: Accuracy results for the specified quality
    """
    model = load_model(checkpoint_path, len(class_names))
    results = test_compressed(model, quality)

    # Print and return results
    print(f"\nEvaluating for Quality Q{quality}:\n")
    for key, acc in results.items():
        print(f"{key}: {acc:.2f}%")

    return results
def resize_image_and_boxes(image, annotations, target_divisor=64):
    w, h = image.size
    new_w = (w // target_divisor) * target_divisor
    new_h = (h // target_divisor) * target_divisor

    resized_image = image.resize((new_w, new_h))

    scale_x = new_w / w
    scale_y = new_h / h

    for ann in annotations:
        bbox = ann['bbox']
        ann['bbox'] = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ]

    return resized_image, annotations
def compress_pillow(image_rgb,types,quality):
    # types in {"jpeg", "webp","jpeg2000"}
    # quality is from 0 to 100, return the image and the bpp
    sizes = image_rgb.size
    temp = io.BytesIO()
    if types in ["jpeg","webp"]:
        image_rgb.save(temp,format = types,quality = quality)
    elif types == "jpeg2000":
        image_rgb.save(temp, format="JPEG2000", quality_mode="rates", quality_layers=[quality])
    else:
        raise ValueError(f"Invalid image type: {types}")
    temp.seek(0)
    filesize = temp.getbuffer().nbytes
    bpp = filesize*8.0/(sizes[0]*sizes[1])
    jpeg_compressed = Image.open(temp)
    return jpeg_compressed,bpp
def process_image(quality):
    image_rgb = kodak_PIL[i]
    jpeg_compressed, bpp = compress_pillow(image_rgb, "jpeg", quality)
    x = kodak_Tensor[i].to(device)
    y = to_tensor(jpeg_compressed).to(device)
    jpeg_metrics[i, quality, 0] = bpp
    jpeg_metrics[i, quality, 1] = SSIM(x, y)
    jpeg_metrics[i, quality, 2] = VIF(x, y)
    jpeg_metrics[i, quality, 3] = PSNR(x, y, max_val=1.0)
    
    print(f"kodim{i}, quality: {quality}, bpp: {bpp}, ssim: {jpeg_metrics[i, quality, 1]}, vif: {jpeg_metrics[i, quality, 2]}, psnr: {jpeg_metrics[i, quality, 3]} ")
    del x, y  # free memory
    torch.cuda.empty_cache()

def BPP(image_tensor,model, num_pixels):
    output = model(image_tensor)
    # Calculate BPP
    bpp = sum(
        torch.log(likelihoods).sum() / (-np.log(2) * num_pixels)
        for likelihoods in output["likelihoods"].values()
    )
    return bpp.item()
def process_image(i):
    model_name,quality = x
    model_class = models[model_name]
    model = model_class(quality = quality, pretrained = True).to(device)
    img = kodak_Tensor[i].to(device)
    img_comp = model(img)["x_hat"].to(device)
    img_comp = img_comp.clamp(0, 1)
    num_pixels = img_comp.shape[0] * img_comp.shape[-1] * img_comp.shape[-2]
    if model_name == "cheng2020-anchor":
        cheng2020_anchor_metrics[i, quality, 0] = BPP(img,model, num_pixels)
        cheng2020_anchor_metrics[i, quality, 1] = SSIM(img, img_comp)
        cheng2020_anchor_metrics[i, quality, 2] = VIF(img, img_comp)
        cheng2020_anchor_metrics[i, quality, 3] = PSNR(img, img_comp, max_val=1.0)
        print(f"model: {model_name}, kodim{i}, quality: {quality}, bpp: {cheng2020_anchor_metrics[i, quality, 0]}, ssim: {cheng2020_anchor_metrics[i, quality, 1]}, vif: {cheng2020_anchor_metrics[i, quality, 2]}, psnr: {cheng2020_anchor_metrics[i, quality, 3]} ")
    elif model_name == "cheng2020-attn":
        cheng2020_attn_metrics[i, quality, 0] = BPP(img,model, num_pixels)
        cheng2020_attn_metrics[i, quality, 1] = SSIM(img, img_comp)
        cheng2020_attn_metrics[i, quality, 2] = VIF(img, img_comp)
        cheng2020_attn_metrics[i, quality, 3] = PSNR(img, img_comp, max_val=1.0)
        print(f"model: {model_name}, kodim{i}, quality: {quality}, bpp: {cheng2020_attn_metrics[i, quality, 0]}, ssim: {cheng2020_attn_metrics[i, quality, 1]}, vif: {cheng2020_attn_metrics[i, quality, 2]}, psnr: {cheng2020_attn_metrics[i, quality, 3]} ")
    del img_comp, img, model  # free memory
    torch.cuda.empty_cache()