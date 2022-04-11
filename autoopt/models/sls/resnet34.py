from autoopt.models.sls.resnet import ResNet


class ResNet34(ResNet):
    def __init__(self, num_classes: int):
        super(ResNet34, self).__init__([3, 4, 6, 3], num_classes)
