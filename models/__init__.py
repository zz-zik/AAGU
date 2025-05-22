from .backbones import BackBones
from .dfine import DFINE, HungarianMatcher
from .criterion import DetCriterion


def build_model(cfg, training=False):
    model = DFINE(cfg)
    if not training:
        return model

    matcher = HungarianMatcher(cfg.criterion.matcher.weight_dict.to_dict(),
                               cfg.criterion.matcher.alpha,
                               cfg.criterion.matcher.gamma
                               )
    losses = DetCriterion(cfg.criterion.losses,
                          cfg.criterion.weight_dict.to_dict(),
                          cfg.model.num_classes,
                          cfg.criterion.alpha,
                          cfg.criterion.gamma,
                          box_fmt='xyxy',
                          matcher=matcher,
                          )

    return model, losses
