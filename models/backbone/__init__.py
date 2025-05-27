from .hgnetv2 import HGNetv2
from .resnet import ResNet50
from .swint import SwinT


def build_backbone(backbone_name: str='hgnetv2', name="B5", use_lab: bool=False, img_size: tuple = (512, 640), return_idx:  list=[1, 2, 3], pretrained: bool=True):
    backbone = None
    if backbone_name == "hgnetv2":
        backbone = HGNetv2(
            name="B5",
            use_lab=use_lab,
            return_idx=return_idx,
            pretrained=pretrained
        )
    elif backbone_name == "resnet50":
        backbone = ResNet50(
            return_idx=return_idx,
            pretrained=pretrained
        )
    elif backbone_name == "swint":
        backbone = SwinT(
            name=name,
            img_size=img_size,
            return_idx=return_idx,
            pretrained=pretrained
        )

    return backbone
