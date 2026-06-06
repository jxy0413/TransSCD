from torchvision import models


def build_resnet34(pretrained=True):
    """Build ResNet-34 across torchvision API versions."""
    if hasattr(models, "ResNet34_Weights"):
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        return models.resnet34(weights=weights)
    return models.resnet34(pretrained=pretrained)

