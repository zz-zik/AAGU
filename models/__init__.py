from .backbone import build_backbone
from .neck import Fusion, ABAMAlignmentLoss
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

    offset_losses = ABAMAlignmentLoss(
        offset_weight=cfg.criterion.abam.offset_weight,
        confidence_weight=cfg.criterion.abam.confidence_weight,
        alignment_weight=cfg.criterion.abam.alignment_weight,
        balance_weight=cfg.criterion.abam.balance_weight,
        smoothness_weight=cfg.criterion.abam.smoothness_weight,
        target_guided_weight=cfg.criterion.abam.target_guided_weight,
        class_aware_weight=cfg.criterion.abam.class_aware_weight,
        target_alignment_ratio=cfg.criterion.abam.target_alignment_ratio,
        modality_bias=cfg.criterion.abam.modality_bias,
        use_target_guidance=cfg.criterion.abam.use_target_guidance
    )

    # offset_losses = ABAMAlignmentLoss(
    #         deformable_alignment_weight=3.0,
    #         boundary_enhancement_weight=2.5,
    #         complementary_fusion_weight=4.0,  # 最重要
    #         fusion_quality_weight=4.5,       # IoU直接优化
    #         alignment_confidence_weight=2.0,
    #         modal_consistency_weight=1.5,
    #         feature_coherence_weight=1.0,
    #         alignment_threshold=0.65,
    #         use_target_guidance=True,
    #         temperature=3.0
    # )

    return model, (losses, offset_losses)
