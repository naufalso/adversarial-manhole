import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from torchvision.transforms import ToTensor, Resize, InterpolationMode
from adv_manhole.models.model_mde import ModelMDE
from adv_manhole.models.monodepth2.networks import ResnetEncoderMatching, DepthDecoder, PoseDecoder, ResnetEncoder
from adv_manhole.models.monodepth2.networks.layers import transformation_from_parameters

from typing import Tuple, List, Callable

class ManyDepth(ModelMDE):
    def __init__(self, model_name, device=None, **kwargs):
        super(ManyDepth, self).__init__(model_name, device=device, **kwargs)

    @staticmethod
    def get_supported_models() -> List[str]:
        return ["kitti_640x192"]

    def _load_and_preprocess_intrinsics(self, intrinsics_path, resize_width, resize_height):
        K = np.eye(4)
        with open(intrinsics_path, 'r') as f:
            K[:3, :3] = np.array(json.load(f))
    
        # Convert normalised intrinsics to 1/4 size unnormalised intrinsics.
        # (The cost volume construction expects the intrinsics corresponding to 1/4 size images)
        K[0, :] *= resize_width // 4
        K[1, :] *= resize_height // 4
    
        invK = torch.Tensor(np.linalg.pinv(K)).unsqueeze(0)
        K = torch.Tensor(K).unsqueeze(0)
    
        if torch.cuda.is_available():
            return K.cuda(), invK.cuda()
        return K, invK
    
    def load(
        self, model_name, model_path:str=None, device=None, instrinsic_json_path:str='', **kwargs
    ) -> Tuple[Callable, int, int]:
        #print(instrinsic_json_path)
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

        if model_name == "kitti_640x192":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "weights")
            
            encoder_path = os.path.join(model_path, model_name, "encoder.pth")
            depth_decoder_path = os.path.join(model_path, model_name, "depth.pth")
            pose_encoder_path = os.path.join(model_path, model_name, "pose_encoder.pth")
            pose_decoder_path = os.path.join(model_path, model_name, "pose.pth")
            
            # Load Encoder
            encoder_dict = torch.load(encoder_path, map_location='cpu')
            #print(encoder_dict.keys())
            self.encoder = ResnetEncoderMatching(
                18, 
                False,
                input_width=encoder_dict['width'],
                 input_height=encoder_dict['height'],
                 adaptive_bins=True,
                 min_depth_bin=encoder_dict['min_depth_bin'],
                 max_depth_bin=encoder_dict['max_depth_bin'],
                 depth_binning='linear',
                 num_depth_bins=96
            )
            filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered_dict_enc)

            # Load depth encoder
            self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
            loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
            self.depth_decoder.load_state_dict(loaded_dict)

            pose_enc_dict = torch.load(pose_encoder_path, map_location='cpu')
            pose_dec_dict = torch.load(pose_decoder_path, map_location='cpu')
        
            self.pose_enc = ResnetEncoder(18, False, num_input_images=2)
            self.pose_dec = PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        
            self.pose_enc.load_state_dict(pose_enc_dict, strict=True)
            self.pose_dec.load_state_dict(pose_dec_dict, strict=True)

            # Move the model to the device
            if device is not None:
                self.encoder.to(device)
                self.depth_decoder.to(device)
                self.pose_enc.to(device)
                self.pose_dec.to(device)

            input_height = encoder_dict["height"]
            input_width = encoder_dict['width']
            
            # Load Intrinsics
            K, invK = self._load_and_preprocess_intrinsics(
                instrinsic_json_path,
                resize_width=input_height,
                resize_height=input_width
            )

            #model = lambda tensor_images: self.depth_decoder(
                #self.encoder(tensor_images)[0]
            #)

            return self.encoder, self.depth_decoder, input_height, input_width, encoder_dict['min_depth_bin'], encoder_dict['max_depth_bin'], K, invK
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

        # Resize the image to the DH input size
        input_image = Resize(
            (self.input_height, self.input_width),
            interpolation=InterpolationMode.BILINEAR,
        )(input_image)

        input_image = input_image.to(self.device)

        return input_image

    def predict(self, tensor_image, source_image, original_shape, return_raw=False, **kwargs):
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

        # Get Pose First
        pose_inputs = [source_image, tensor_image]
        pose_inputs = [self.pose_enc(torch.cat(pose_inputs, 1))]
        axisangle, translation = self.pose_dec(pose_inputs)
        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

        # Mono
        pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
        source_image *= 0

        output, lowest_cost, _ = self.encoder(
            current_image=tensor_image,
            lookup_images=source_image.unsqueeze(1),
            poses=pose.unsqueeze(1),
            K=self.K,
            invK=self.invK,
            min_depth_bin=self.min_depth_bin,
            max_depth_bin=self.max_depth_bin
        )

        disparity = self.depth_decoder(output)[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disparity, original_shape, mode="bilinear", align_corners=False
        )

        return disp_resized
            
    def visualize_prediction(self, prediction, **kwargs):
        """
        Visualizes the disparity prediction.

        Args:
            prediction (torch.Tensor): The disparity prediction.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: The visualized disparity prediction.

        """
        disp_resized_np = prediction.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)

        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        
        return colormapped_im

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
            