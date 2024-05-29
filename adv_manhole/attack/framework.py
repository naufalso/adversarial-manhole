import torch

from adv_manhole.models.model_mde import ModelMDE
from adv_manhole.models.model_ss import ModelSS
from adv_manhole.attack.losses import AdvManholeLosses


class AdvManholeFramework:

    def __init__(
        self,
        texture_res: int,
        mde_model: ModelMDE,
        ss_model: ModelSS,
        loss: AdvManholeLosses,
    ):
        self.texture_res = texture_res
        self.mde_model = mde_model
        self.ss_model = ss_model
        self.loss = loss
