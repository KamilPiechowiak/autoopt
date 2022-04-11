from autoopt.models.small.small_resnet import SmallResnet


class SmallResnet20(SmallResnet):
    def __init__(self, num_classes: int) -> None:
        super(SmallResnet20, self).__init__(20, num_classes)
