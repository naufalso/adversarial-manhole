import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from torchvision.transforms import ToTensor, Resize, InterpolationMode
from adv_manhole.models.model_ss import ModelSS

from typing import Tuple, List, Callable
from mmseg.visualization import SegLocalVisualizer
from mmseg.apis import init_model, inference_model

class MMSegmentation(ModelSS):
    def __init__(self, model_name, device=None, **kwargs):
        super(MMSegmentation, self).__init__(model_name, device=device, **kwargs)

    @staticmethod
    def get_supported_models() -> List[str]:
        return [
            "pspnet_r50-d8_4xb2-80k_cityscapes-512x1024", 
            'ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024',
            'bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024',
            'icnet_r18-d8_4xb2-160k_cityscapes-832x832'
        ]

    def _get_model_map(self) -> dict:
        model_files_map_dict = {
            "pspnet_r50-d8_4xb2-80k_cityscapes-512x1024" : "pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth",
            "ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024" : "ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230425_162633-81601db0.pth",
            "bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024" : "bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth",
            "icnet_r18-d8_4xb2-160k_cityscapes-832x832": "icnet_r18-d8_832x832_160k_cityscapes_20210925_230153-2c6eb6e0.pth"
        }

        return model_files_map_dict
    
    def load(
        self, model_name, model_path:str=None, device=None, **kwargs
    ) -> Tuple[Callable, int, int]:
        """
        Load a pre-trained model for monocular depth estimation.

        Args:
            model_name (str): Name of the model to load.
            model_path (str, optional): Path to the model weights. If not provided, the default path will be used.
            device (str, optional): Device to load the model on. If not provided, the default device will be used.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Callable, int, int]: The loaded model, input height, and input width.

        Raises:
            ValueError: If the specified model is not supported.

        """
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "weights")

        if model_name in self._get_model_map().keys():
        #if model_name == "pspnet_r50-d8_4xb2-80k_cityscapes-512x1024":
            # Get Filename for model weight
            weight_filename = self._get_model_map()[model_name]
            
            model_path_weight = os.path.join(model_path, weight_filename)
            model_path_config = os.path.join(model_path, model_name + ".py")

            model = init_model(model_path_config, model_path_weight, device)
            height, width = model.cfg['crop_size']

            return model, height, width

    def preprocess(self, input_image, **kwargs):
        """
        Preprocesses the input image for the monodepth2 model.

        Args:
            input_image (PIL.Image.Image or torch.Tensor): The input image to be preprocessed.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """

        pass

    def predict(self, tensor_images, original_shape, return_raw=False, **kwargs):
        """
        Predicts the output of the model for the given input tensor_images.

        Args:
            tensor_images (torch.Tensor): Input tensor of images.
            original_shape (tuple): The original shape of the input images.
            return_raw (bool): Whether to return the raw output of the model.

        Returns:
            torch.Tensor: Output tensor of the model predictions.
        """
        
        return inference_model(self.model, tensor_images)

    def visualize_prediction(self, prediction, **kwargs):
        """
        Visualizes the disparity prediction.

        Args:
            prediction (torch.Tensor): The disparity prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: The visualized disparity prediction.

        """
        pass

    def plot(self, image, prediction, **kwargs):
        """
        Plots the input image and the disparity prediction.

        Args:
            image (numpy.ndarray): The input image.
            prediction (torch.Tensor): The disparity prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        seg_local_visualizer = SegLocalVisualizer(
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir=os.path.dirname(os.path.abspath(__file__))
        )

        out_file = "test"
        seg_local_visualizer.add_datasample(out_file, image, prediction, show=False, with_labels=False)

if __name__ == "__main__":
    # Test check the current path
    print(os.path.dirname(os.path.abspath(__file__)))