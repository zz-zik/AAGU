from .hgnetv2 import HGNetv2
from .resnet import ResNet50
from .swint import SwinT


def build_backbone(cfg):
    backbone = None
    if cfg.model.backbone_name == "hgnetv2":
        backbone = HGNetv2(name="B0", use_lab=True, return_idx=[0, 1, 2, 3])
    elif cfg.model.backbone_name == "resnet50":
        backbone = ResNet50(cfg)
    elif cfg.model.backbone_name == "swint":
        backbone = SwinT(cfg)

    return backbone
