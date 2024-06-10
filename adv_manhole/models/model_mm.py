from adv_manhole.models.model import Model

class ModelMM(Model):
    type = "mm"

    def __init__(self, model_name, device=None, **kwargs):
        super(ModelMM, self).__init__(model_name, device=device, **kwargs)