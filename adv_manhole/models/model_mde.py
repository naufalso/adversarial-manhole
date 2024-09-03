from adv_manhole.models.model import Model


class ModelMDE(Model):
    type = "mde"

    def __init__(self, model_name, device=None, **kwargs):
        super(ModelMDE, self).__init__(model_name, device=device, **kwargs)
