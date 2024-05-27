from adv_manhole.models.model import Model


class ModelSS(Model):
    type = "ss"

    def __init__(self, model_name, device=None, **kwargs):
        super(ModelSS, self).__init__(model_name, device=device, **kwargs)
