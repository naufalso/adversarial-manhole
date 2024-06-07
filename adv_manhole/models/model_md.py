from adv_manhole.models.model import Model

class ModelMD(Model):
    type = "dh"

    def __init__(self, model_name, device=None, **kwargs):
        super(ModelMD, self).__init__(model_name, device=device, **kwargs)