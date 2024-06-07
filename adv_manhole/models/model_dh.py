from adv_manhole.models.model import Model

class ModelDH(Model):
    type = "dh"

    def __init__(self, model_name, device=None, **kwargs):
        super(ModelDH, self).__init__(model_name, device=device, **kwargs)