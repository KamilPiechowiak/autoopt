from autoopt.losses.classification_loss_creator import ClassificationLossCreator
from autoopt.losses.vae_loss_creator import VAELossCreator


class LossCreatorsFactory:
    def __init__(self) -> None:
        pass

    def get_loss_creator(self, task):
        if task == 'classification':
            return ClassificationLossCreator()
        elif task == 'vae':
            return VAELossCreator()
