import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor, Resize, InterpolationMode
from adv_manhole.models.monodepth2.networks import ResnetEncoder, DepthDecoder
from adv_manhole.models.model_mde import MDEModel
from typing import Tuple, List, Callable


class MonoDepth2(MDEModel):
    def __init__(self, device=None, **kwargs):
        super(MonoDepth2, self).__init__(device=device, **kwargs)
        # Add your model layers and configurations here

    def get_supported_models(self) -> List[str]:
        """
        Get the list of supported models for monocular depth estimation.

        Returns:
            list: The list of supported models.
        """
        return ["mono_640x192"]

    def load(
        self, model_name, model_path=None, device=None, **kwargs
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
            base_model_path = os.path.join(current_dir, "monodepth2", "weights")

        if model_name == "mono_640x192":
            # Initialize the model
            self.encoder = ResnetEncoder(18, False)
            self.depth_decoder = DepthDecoder(self.encoder.num_ch_enc)

            encoder_path = os.path.join(base_model_path, model_name, "encoder.pth")
            depth_decoder_path = os.path.join(base_model_path, model_name, "depth.pth")

            # Load the encoder weights
            loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
            filtered_dict_enc = {
                k: v
                for k, v in loaded_dict_enc.items()
                if k in self.encoder.state_dict()
            }
            self.encoder.load_state_dict(filtered_dict_enc)

            # Load the depth decoder weights
            loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
            self.depth_decoder.load_state_dict(loaded_dict)

            # Set the model to evaluation mode
            self.encoder.eval()
            self.depth_decoder.eval()

            # Move the model to the device
            if device is not None:
                self.encoder.to(device)
                self.depth_decoder.to(device)

            input_height = loaded_dict_enc["height"]
            input_width = loaded_dict_enc["width"]

            model = lambda tensor_images: self.depth_decoder(
                self.encoder(tensor_images)
            )

            return model, input_height, input_width

        else:
            raise ValueError(f"Model {model_name} is not supported yet.")

    def preprocess(self, input_image, **kwargs):
        """
        Preprocesses the input image for the monodepth2 model.

        Args:
            input_image (PIL.Image.Image or torch.Tensor): The input image to be preprocessed.

        Returns:
            torch.Tensor: The preprocessed image tensor.
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
        if return_raw:
            features = self.encoder(tensor_images)
            outputs = self.depth_decoder(features)

            return features, outputs

        disparity = self.model(tensor_images)[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disparity, original_shape, mode="bilinear", align_corners=False
        )

        return disp_resized

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

        # PLOTTING
        disp_resized_np = prediction.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)

        fig = plt.figure(figsize=(5, 5))
        plt.subplot(211)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(212)
        plt.imshow(disp_resized_np, cmap="magma", vmax=vmax)
        plt.title("Disparity")
        plt.axis("off")

        return fig


if __name__ == "__main__":
    # Test check the current path
    print(os.path.dirname(os.path.abspath(__file__)))
