from adv_manhole.models.monodepth2.monodepth2_model import MonoDepth2
from adv_manhole.models.supergradients.supergradients_model import SuperGradientsModel
from adv_manhole.models.depth_hints.depth_hints_model import DepthHints
from typing import Optional
from enum import Enum

# Create ENUM for model types
ModelType = Enum("ModelType", ["MDE", "SS", "DH", "MD"])


def load_models(
    model_type: ModelType, model_name: str, device: Optional[str] = None, **kwargs
):
    """
    Get the specified model.

    Args:
        model_type (ModelType): The type of the model to get.
        model_name (str): The name of the model to get.
        device (str, optional): The device to load the model on. If not provided, the default device will be used.
        **kwargs: Additional keyword arguments to be passed to the model loading process.

    Returns:
        Model: The specified model.

    Raises:
        ValueError: If the specified model type is not supported.
    """
    if model_type == ModelType.MDE:
        if model_name in MonoDepth2.get_supported_models():
            return MonoDepth2(model_name, device=device, **kwargs)
    elif model_type == ModelType.SS:
        if model_name in SuperGradientsModel.get_supported_models():
            return SuperGradientsModel(model_name, device=device, **kwargs)
    elif model_type == ModelType.DH:
        if model_name in DepthHintsModel.get_supported_models():
            return DepthHintsModel(model_name, device=device, **kwargs)
    elif model_type == ModelType.MD:
        if model_name in ManyDepthModel.get_supported_models():
            return ManyDepthModel(model_name, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types are 'mde' and 'ss'."
        )
