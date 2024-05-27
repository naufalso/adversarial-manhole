import torch
import numpy as np
import matplotlib.pyplot as plt

from adv_manhole.models.model_ss import ModelSS
from super_gradients.training import models
from torchvision.transforms import ToTensor, Resize, Normalize, InterpolationMode
from typing import Tuple, List, Callable
from PIL import Image


class SuperGradientsModel(ModelSS):
    type = "model"

    def __init__(self, model_name, device=None, **kwargs):
        super(SuperGradientsModel, self).__init__(model_name, device=device, **kwargs)

    def get_supported_models(self) -> List[str]:
        """
        Get the list of supported models.

        Returns:
            list: The list of supported models.
        """
        return ["ddrnet_23", "ddrnet_23_slim", "ddrnet_39"]

    def load(
        self, model_name, model_path=None, device=None, **kwargs
    ) -> Tuple[Callable, int, int]:
        """
        Load a pre-trained model.

        Args:
            model_name (str): The name of the model to load.
            model_path (str, optional): The path to the saved model file. If not provided, the default path will be used.
            device (str, optional): The device to load the model on. If not provided, the default device will be used.
            **kwargs: Additional keyword arguments to be passed to the model loading process.

        Returns:
            Tuple[Callable, int, int]: The loaded model, input height, and input width.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            ValueError: If the specified model name is not supported.
            Exception: If any other error occurs during the model loading process.
        """

        # Load the model
        model = models.get(
            model_name=model_name,
            pretrained_weights="cityscapes",
        )

        # Set the model to evaluation mode
        model.eval()

        # Move the model to the specified device
        if device is not None:
            model.to(device)

        if "ddrnet" in model_name:
            input_height = 1024
            input_width = 2048
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        return model, input_height, input_width

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

        # Normalize the image
        input_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            input_image
        )

        input_image = input_image.to(self.device)

    def predict(self, tensor_images, original_shape, **kwargs):
        """
        Perform semantic segmentation predictions on the given tensor images.

        Args:
            tensor_images (torch.Tensor): Input tensor images to perform predictions on.
            original_shape (tuple): The original shape of the input images before resizing.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Predicted probabilities of the semantic segmentation classes.
        """
        logits = self.model(tensor_images)
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
        pred_labels = torch.argmax(prediction, dim=1)

        # Convert the predicted labels to a numpy array
        pred_labels_np = pred_labels.squeeze().detach().cpu().numpy()

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
        plt.title("Input", fontsize=22)
        plt.axis("off")

        plt.subplot(212)
        plt.imshow(pred_labels_colored)
        plt.title("Segmented Image", fontsize=22)
        plt.axis("off")
        plt.tight_layout()

        return fig
