import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from PIL import Image

from torchvision.transforms import ToTensor, Resize, InterpolationMode
from adv_manhole.models.model_ss import ModelSS

from typing import Tuple, List, Callable
# from mmseg.visualization import SegLocalVisualizer
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
            model.eval()
            height, width = model.cfg['crop_size']

            return model, height, width

    def preprocess(self, input_image, **kwargs):
        """
        Preprocesses the input image before feeding it into the model.

        Args:
            input_image (numpy.ndarray): The input image to be preprocessed.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The preprocessed image.

        """
        # Check if the image is not a tensor
        if not torch.is_tensor(input_image):
            # Convert the image to a tensor
            input_image = ToTensor()(input_image).unsqueeze(0)

        # Resize the image to the MDE input size
        input_image = Resize(
            (self.input_height, self.input_width),
            interpolation=InterpolationMode.BILINEAR,
        )(input_image)

        input_image = input_image.to(self.device)

        input_image = self.model.data_preprocessor({"inputs": input_image * 255.0}, training=False)['inputs']

        return input_image

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
        
        logits = self.model(tensor_images, mode='tensor')
        logits_reshape = torch.nn.functional.interpolate(
            logits, size=original_shape, mode="bilinear", align_corners=False
        )
        probs = torch.softmax(logits_reshape, dim=1)
        return probs

    def get_class_names(self):
        """
        Get the class names for cityscapes semantic segmentation.
        TODO: Move this to a separate class.

        Returns:
            list: The list of class names.
        """

        return [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

    def _cityscapes_color_map(self):
        """
        Returns the color map used in the Cityscapes dataset.
        TODO: Move this to a separate class.

        The color map is a list of RGB tuples representing the colors used to label different classes in the dataset.

        Returns:
            list: A list of RGB tuples representing the color map.
        """
        color_maps = [
            (128, 64, 128),
            (244, 35, 232),
            (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (220, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 70),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
        ]
        return np.array(color_maps, dtype=np.uint8)

    def visualize_prediction(self, prediction, **kwargs):
        """
        Visualizes the predicted labels by applying a color palette to the predicted labels.

        Args:
            prediction (torch.Tensor): The predicted labels.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: The predicted labels with colors applied.

        """
        # Get the predicted class labels
        pred_labels = torch.argmax(prediction, dim=0)

        # Convert the predicted labels to a numpy array
        pred_labels_np = pred_labels.detach().cpu().numpy()

        # Resize the predicted labels to the original image size
        pred_labels_np = np.array(
            Image.fromarray(pred_labels_np.astype(np.uint8))
        )  # .resize(size=cityscape_img.size, resample=Image.NEAREST))

        # Create a color pallette, selecting a color for each class
        color_palette = self._cityscapes_color_map()

        # Apply the color pallette to the predicted labels
        pred_labels_colored = color_palette[pred_labels_np]

        return pred_labels_colored


    def plot(self, image, prediction, **kwargs):
        """
        Plots the prediction.

        Args:
            image: The input image.
            prediction: The prediction to be plotted.
            **kwargs: Additional keyword arguments for customizing the plot.

        Returns:
            Figure: The plot figure.
        """
        # Get the predicted class labels
        pred_labels = torch.argmax(prediction, dim=0)

        # Convert the predicted labels to a numpy array
        pred_labels_np = pred_labels.detach().cpu().numpy()

        # Resize the predicted labels to the original image size
        pred_labels_np = np.array(
            Image.fromarray(pred_labels_np.astype(np.uint8))
        )  # .resize(size=cityscape_img.size, resample=Image.NEAREST))

        # Create a color pallette, selecting a color for each class
        color_palette = self._cityscapes_color_map()

        # Apply the color pallette to the predicted labels
        pred_labels_colored = color_palette[pred_labels_np]

        fig = plt.figure(figsize=(5, 5))
        plt.subplot(211)
        plt.imshow(image)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(212)
        plt.imshow(pred_labels_colored)
        plt.title("Segmented Image")
        plt.axis("off")
        plt.tight_layout()

        return fig

if __name__ == "__main__":
    # Test check the current path
    print(os.path.dirname(os.path.abspath(__file__)))