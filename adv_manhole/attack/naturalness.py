import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights

class ResizeAndNormalization(nn.Module):
    def __init__(self, mean, std, expected_size=(224, 224), device=None):
        super(ResizeAndNormalization, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.expected_size = expected_size

    def forward(self, img):
        # normalize ``img``
        img = F.interpolate(img, size=self.expected_size, mode='bilinear')
        return (img - self.mean) / self.std


class AdvContentLoss(nn.Module):

    def __init__(self, candidate_images, input_size=224, feature_extractor='vgg16', cnn_content_idx=4, device=None):
        super(AdvContentLoss, self).__init__()
        
        self.candidate_images = candidate_images
        self.candidate_count = candidate_images.shape[0]

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the feature extractor
        if feature_extractor == 'vgg16':
            self.model = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(self.device)
        else:
            # TODO: Add support for other feature extractors
            raise ValueError('Feature extractor not supported')
        
        self.feature_extractor = self._build_feature_extractor(self.model, input_size, cnn_content_idx).to(self.device)
        self.candidate_features = self.feature_extractor(candidate_images).detach().view(self.candidate_count, -1) 

    def _build_feature_extractor(self, model, input_size, last_conv_layer_index):
        features = []

        # Add normalization
        resize_normalization = ResizeAndNormalization(torch.tensor([0.485, 0.456, 0.406], device=self.device), torch.tensor([0.229, 0.224, 0.225], device=self.device), (input_size, input_size))
        features.append(resize_normalization)

        i = 0
        for layer in model.children():
            features.append(layer)
            
            if isinstance(layer, nn.Conv2d):
                i += 1
                if i == last_conv_layer_index:
                    break

        return nn.Sequential(*features)

    def forward(self, input_images):
        input_features = self.feature_extractor(input_images).view(1, -1).repeat(self.candidate_count, 1)

        mae_losses = torch.mean(torch.abs(input_features - self.candidate_features), dim=1)

        min_loss, min_idx = torch.min(mae_losses, dim=0)

        return min_loss, min_idx