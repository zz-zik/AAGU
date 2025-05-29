from .backbone import build_backbone
from .neck import Fusion
from .dfine import DFINE, HungarianMatcher, DFINECriterion


def build_model(cfg, training=False):
    model = DFINE(cfg)
    if not training:
        return model

    matcher = HungarianMatcher(cfg.criterion.matcher.weight_dict.to_dict(),
                               cfg.criterion.matcher.alpha,
                               cfg.criterion.matcher.gamma
                               )

    losses = DFINECriterion(
        matcher=matcher,
        weight_dict=cfg.criterion.weight_dict.to_dict(),
        losses=cfg.criterion.losses,
        alpha=cfg.criterion.alpha,
        gamma=cfg.criterion.gamma,
        num_classes=cfg.model.num_classes,
        reg_max=cfg.criterion.reg_max,
                          )

    return model, losses
