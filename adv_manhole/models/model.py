import torch
import numpy as np
from PIL import Image

from abc import ABC, abstractmethod
from typing import Tuple, List, Callable


class Model(ABC):
    type = "model"

    model = None
    input_height = None
    input_width = None

    def __init__(self, model_name, device=None, **kwargs):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.input_height, self.input_width = self.load(
            model_name, device=self.device
        )

    def get_input_shape(self, input_image):
        """
        Get the original shape of the input image.

        Args:
            input_image (numpy.ndarray, torch.Tensor, or PIL.Image): The input image.

        Returns:
            tuple: The original shape of the input image.
        """

        # Get the original shape of the input image (H, W)
        if torch.is_tensor(input_image):
            input_shape = input_image.shape[-2:]
        elif isinstance(input_image, Image.Image):
            input_shape = input_image.size[::-1]
        elif isinstance(input_image, np.ndarray):
            # Check if the input has a batch dimension
            if len(input_image.shape) == 3:
                input_shape = input_image.shape[:2]
            else:
                input_shape = input_image.shape[-3:-1]

        return input_shape

    @staticmethod
    def get_supported_models() -> List[str]:
        """
        Get the list of supported models.

        Returns:
            list: The list of supported models.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def preprocess(self, input_image, **kwargs):
        """
        Preprocesses the input image before feeding it into the model.

        Args:
            input_image (numpy.ndarray): The input image to be preprocessed.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The preprocessed image.

        """
        pass

    @abstractmethod
    def predict(self, tensor_images, original_shape, **kwargs):
        """
        Predicts the output for the given tensor images.

        Args:
            tensor_images (Tensor): The input tensor images.
            original_shape (tuple): The original shape of the input images.

        Returns:
            Tensor: The predicted output tensor.

        Raises:
            ValueError: If the input tensor images are invalid.

        Examples:
            >>> model = Model()
            >>> tensor_images = torch.rand(1, 3, 224, 224)
            >>> original_shape = (224, 224)
            >>> output = model.predict(tensor_images, original_shape)
        """
        pass

    def __call__(self, input_image, **kwargs):
        """
        Perform the prediction on the input image.

        Args:
            input_image (numpy.ndarray or tensor): The input image.

        Returns:
            Tensor: The predicted output tensor.
        """

        # Get the shape of the input image
        input_shape = self.get_input_shape(input_image)

        # Preprocess the input image
        tensor_images = self.preprocess(input_image)

        # Perform the prediction
        prediction = self.predict(tensor_images, input_shape)

        return prediction

    @abstractmethod
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
        pass
